# Metal Kernel Design: Fused rANS Decode + GEMM

**Status:** Design Document  
**Target:** Apple Silicon (M1/M2/M3/M4)  
**Goal:** Fuse entropy decoding with matrix multiplication to minimize memory traffic

---

## Overview

Standard LLM inference:
```
Memory → Dequantize → GEMM → Output
```

Our approach:
```
Memory (compressed) → Decode+Dequantize → GEMM → Output
                      ↑
              Fused in registers, never write to memory
```

Key insight: Decode directly into registers, multiply immediately, never materialize full weight matrix.

---

## Architecture

### Tile-Based Processing

```
Weight Matrix W (M × N):
┌─────────────────────────────────────┐
│  Tile[0,0]  │  Tile[0,1]  │  ...    │  Each tile: 64×64 or 128×32
│─────────────│─────────────│─────────│
│  Tile[1,0]  │  Tile[1,1]  │  ...    │  Independent rANS streams
│─────────────│─────────────│─────────│
│    ...      │    ...      │  ...    │
└─────────────────────────────────────┘
```

Each tile has:
- Independent frequency table (fits in shared memory)
- Independent compressed bitstream
- Can decode in parallel

### Memory Layout

```cpp
struct CompressedTile {
    uint16_t freq[16];        // Frequency table (32 bytes)
    uint16_t cumfreq[17];     // Cumulative frequencies (34 bytes)
    uint32_t stream_offset;   // Offset into compressed data
    uint32_t n_symbols;       // Number of symbols in tile
};

struct CompressedLayer {
    float scale;              // Dequantization scale
    float zero_point;         // Dequantization offset
    uint32_t n_tiles_m;       // Tiles in M dimension
    uint32_t n_tiles_n;       // Tiles in N dimension
    CompressedTile tiles[];   // Tile metadata
    uint8_t data[];           // Compressed bitstreams
};
```

---

## Kernel Design

### Kernel 1: Fused Decode + GEMV (Single Token)

For autoregressive generation (batch=1):

```metal
kernel void fused_rans_gemv(
    device const CompressedLayer* layer [[buffer(0)]],
    device const float* input          [[buffer(1)]],  // (N,)
    device float* output               [[buffer(2)]],  // (M,)
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Each threadgroup handles one output row
    uint row = tgid;
    
    // Shared memory for frequency table
    threadgroup uint16_t freq[16];
    threadgroup uint16_t cumfreq[17];
    threadgroup uint8_t sym_table[4096];  // PROB_SCALE
    
    // Load frequency table (first 16 threads)
    if (simd_lane < 16) {
        freq[simd_lane] = layer->tiles[row].freq[simd_lane];
    }
    // ... build cumfreq and sym_table
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Initialize rANS state
    uint32_t state = load_initial_state(layer, row);
    uint ptr = layer->tiles[row].stream_offset + 4;
    
    // Accumulator for dot product
    float acc = 0.0f;
    
    // Decode and multiply in chunks
    for (uint col = simd_lane; col < N; col += 32) {
        // Decode one symbol
        uint slot = state & 0xFFF;  // PROB_SCALE - 1
        uint8_t sym = sym_table[slot];
        
        // Update state
        uint freq_s = freq[sym];
        uint cumfreq_s = cumfreq[sym];
        state = freq_s * (state >> 12) + slot - cumfreq_s;
        
        // Renormalize
        while (state < (1 << 23)) {
            state = (state << 8) | load_byte(layer->data, ptr++);
        }
        
        // Dequantize
        float weight = float(sym) * layer->scale + layer->zero_point;
        
        // Multiply and accumulate
        acc += weight * input[col];
    }
    
    // SIMD reduction
    acc = simd_sum(acc);
    
    if (simd_lane == 0) {
        output[row] = acc;
    }
}
```

### Kernel 2: Batched Decode for Prefill

For prompt processing (batch > 1):

```metal
kernel void fused_rans_gemm(
    device const CompressedLayer* layer [[buffer(0)]],
    device const float* input          [[buffer(1)]],  // (B, N)
    device float* output               [[buffer(2)]],  // (B, M)
    constant uint& batch_size          [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // 2D decomposition: each threadgroup handles a tile of output
    // Decode weights once, reuse across batch dimension
    
    uint tile_row = tgid.y;
    uint tile_col = tgid.x;
    
    // Load and decode tile into shared memory
    threadgroup float tile_weights[TILE_M][TILE_N];
    
    // ... decode rANS stream into tile_weights ...
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Standard tiled GEMM from here
    // ... matrix multiply using decoded weights ...
}
```

---

## Optimization Strategies

### 1. Interleaved rANS Streams

Problem: Single rANS stream is sequential.
Solution: Interleave 4-8 streams, decode in parallel.

```
Stream 0: symbols 0, 4, 8, 12, ...
Stream 1: symbols 1, 5, 9, 13, ...
Stream 2: symbols 2, 6, 10, 14, ...
Stream 3: symbols 3, 7, 11, 15, ...
```

Each SIMD lane handles one stream.

### 2. Vectorized State Updates

```metal
// Process 4 symbols at once using SIMD
simd_uint4 states = load_states();
simd_uint4 slots = states & 0xFFF;
simd_uchar4 syms = lookup_symbols(slots);
states = update_states(states, syms);
```

### 3. Prefetch Compressed Data

```metal
// Prefetch next chunk while processing current
device const uint8_t* next_chunk = layer->data + ptr + 64;
simdgroup_async_copy(prefetch_buffer, next_chunk, 64);
```

### 4. Shared Memory Frequency Table

The 16-symbol frequency table (32 bytes) fits easily in shared memory.
The 4096-entry symbol lookup table (4KB) also fits.

---

## Performance Analysis

### Compute Cost

Per symbol:
- Table lookup: 1 op
- State update: 4 ops (mul, shift, add, sub)
- Renormalization: ~1 op amortized
- Dequantize: 2 ops (mul, add)
- MAC: 2 ops

Total: ~10 ops per symbol

### Throughput Calculation

M3 Pro GPU specs:
- 7 TFLOPS (FP32)
- ~14 trillion ops/sec (including INT)

At 10 ops/symbol:
- **1.4 trillion symbols/sec theoretical**
- With memory/control overhead: ~100 billion symbols/sec realistic

For 7B model:
- Decode time: 7B / 100B = **0.07 ms**
- Bandwidth saved: (4-3.5)/4 × 3.5GB = 0.44 GB
- Time saved at 150 GB/s: 0.44/150 × 1000 = **2.9 ms**

**Net gain: 2.9 - 0.07 = 2.83 ms per forward pass**

---

## Comparison to Standard 4-bit GEMM

| Metric | Standard 4-bit | Entropy-Coded |
|--------|---------------|---------------|
| Memory read | 3.5 GB | 2.4 GB |
| Decode ops | 0 | 70B |
| Time (bandwidth) | 23.3 ms | 16.0 ms |
| Time (decode) | 0 | 0.07 ms |
| **Total** | **23.3 ms** | **16.1 ms** |
| **Speedup** | — | **1.45x** |

---

## Implementation Roadmap

### Phase 1: Reference Implementation
- [ ] Python rANS codec (done)
- [ ] NumPy-based fused decode+GEMV
- [ ] Correctness tests

### Phase 2: Metal Prototype
- [ ] Basic Metal rANS decode kernel
- [ ] Shared memory frequency tables
- [ ] Single-stream decode + GEMV

### Phase 3: Optimization
- [ ] Interleaved streams
- [ ] SIMD vectorization
- [ ] Memory coalescing
- [ ] Occupancy tuning

### Phase 4: MLX Integration
- [ ] Custom MLX operation
- [ ] Compressed weight format
- [ ] End-to-end inference

---

## References

1. Giesen, F. "Interleaved entropy coders" (2014)
2. Duda, J. "Asymmetric numeral systems" (2014)
3. Apple Metal Best Practices Guide
4. MLX Documentation - Custom Operations
