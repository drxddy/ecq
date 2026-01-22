# Hybrid Approach: Combining ASTC + rANS for Optimal Compression

## Understanding ASTC's "Zero Overhead" Magic

From Apple's paper, ASTC works because:

1. **Fixed-function decoder** - Baked into GPU's texture sampling unit
2. **Block-based** - 6×6 blocks = 36 weights → 128 bits (3.56 bits/weight)
3. **Transparent to shaders** - Just sample a texture, hardware does the rest
4. **Fused min-add** - Per-block minimum stored separately, added during matmul

```
Memory → [Texture Unit] → Decoded Values → [Shader/Compute]
              ↑
        Hardware decode
         (no cycles!)
```

## Why We Can't Just Use ASTC

| Issue | ASTC Limitation | Our rANS |
|-------|-----------------|----------|
| Compression | Fixed 3.56 bits/weight | Adaptive 1.5-2.2 bits |
| Quality | Lossy (needs LoRA recovery) | Lossless |
| Block overhead | 128 bits minimum per block | Stream-based, no minimum |
| Flexibility | HDR-ch mode only | Any probability distribution |

## Hybrid Approaches

### Approach 1: Two-Stage Coding (ASTC + rANS residual)

```
Original weights → [Quantize to 4-bit] → [ASTC encode] → 3.56 bits (HW decode)
                                              ↓
                                        [Residual]
                                              ↓
                                    [rANS encode residual] → 0.5-1 bits (SW decode)
```

**Problem:** ASTC is lossy, so there's no meaningful "residual" to encode.

### Approach 2: Learned Block Coding (ASTC-inspired, our compression)

Instead of ASTC's fixed codec, use a learned block-based entropy coder:

```
Weights → [Reshape to 6×6 blocks] → [Block-wise rANS] → ~2 bits/weight
```

**Advantage:** Block structure enables parallel decode (one thread per block)
**Disadvantage:** Still software decode, still 29% overhead

### Approach 3: Use Texture Memory for Tables (Recommended)

The key insight: **use hardware texture sampling for our lookup tables**, not the weights themselves.

```cpp
// Current: Table in device memory
uint8_t s = sym_table[slot];  // ~300 cycle latency

// Proposed: Table as 1D texture
uint8_t s = sym_texture.sample(slot / 16384.0);  // Hardware prefetch!
```

**Why this helps:**
- Texture units have dedicated cache hierarchy
- Hardware prefetching optimized for spatial locality
- Our `sym_table[slot]` access pattern is random, but within a small range (16K entries)

### Approach 4: Block-Parallel rANS with Texture-Cached State

Combine block structure with texture memory:

```cpp
// Store per-block frequency tables as texture
texture2d<half> freq_tex;  // 16 frequencies × N blocks

// Each threadgroup decodes one block
kernel void block_rans_decode(
    texture2d<half> freq_tex,      // HW-cached frequency tables
    device uint8_t* compressed,     // Interleaved streams
    ...
) {
    // Sample frequency table from texture (hardware cached)
    half freq = freq_tex.sample(sampler, float2(symbol, block_id));
    
    // Rest of decode...
}
```

### Approach 5: ASTC for Low-Entropy Layers, rANS for High-Entropy

Layer-adaptive approach based on measured entropy:

| Layer Type | Entropy | Approach | Bits/Weight |
|------------|---------|----------|-------------|
| Embedding | ~1.5 bits | rANS | 1.8 |
| Attention Q/K/V | ~1.2 bits | rANS | 1.5 |
| MLP down_proj | ~2.5 bits | ASTC | 3.56 |
| MLP up_proj | ~1.8 bits | rANS | 2.2 |

**Advantage:** Best of both worlds
**Disadvantage:** Two code paths, complex format

---

## Recommended Path Forward

### Phase 1: Reduce rANS Overhead (Current Focus)

Our 29% overhead comes from:

| Source | Contribution | Mitigation |
|--------|--------------|------------|
| Sequential decode | ~15% | More streams, wider interleaving |
| Table lookups | ~8% | Texture memory for sym_table |
| State renormalization | ~6% | Unroll, vectorize |

**Target:** Get overhead below 15%

### Phase 2: Texture-Backed Tables

```cpp
// Encode sym_table as 128×128 texture (16K entries)
texture2d<uint8_t> sym_texture;

// In kernel:
constexpr sampler s(coord::pixel, filter::nearest);
uint8_t sym = sym_texture.read(uint2(slot % 128, slot / 128)).r;
```

### Phase 3: Block-Structured Encoding

For better parallelism:

```
Matrix: (M, N) → Blocks: (M/32, N/32) × (32, 32)

Each 32×32 block:
  - 1024 weights
  - 64 interleaved streams (16 weights each)
  - Independent frequency table
  - One threadgroup
```

**Result:** 64-way parallelism per block, thousands of blocks in flight

---

## Theoretical Limits

| Approach | Bits/Weight | Decode Overhead | Effective Bits |
|----------|-------------|-----------------|----------------|
| ASTC (Apple) | 3.56 | 0% | 3.56 |
| rANS current | 2.17 | 29% | 2.80 |
| rANS + texture tables | 2.17 | 15% (est) | 2.50 |
| Block-parallel rANS | 2.17 | 10% (est) | 2.39 |

**Even with optimizations, we should beat ASTC's 3.56 effective bits.**

---

## Implementation Priority

1. **Texture-backed sym_table** - Easiest, biggest impact on random access
2. **Wider interleaving** - More streams = more parallelism
3. **Block structure** - Independent blocks = better GPU utilization
4. **Layer-adaptive** - Use rANS where it wins, ASTC where it wins

---

## Code Sketch: Texture-Backed Tables

```python
# Encode sym_table as Metal texture
def create_sym_texture(table: np.ndarray) -> Metal.MTLTexture:
    desc = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
        Metal.MTLPixelFormatR8Uint,  # 8-bit unsigned int
        128, 128,  # 16K entries
        False
    )
    texture = device.newTextureWithDescriptor_(desc)
    texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
        Metal.MTLRegionMake2D(0, 0, 128, 128),
        0,
        table.tobytes(),
        128
    )
    return texture
```

```metal
// In kernel
texture2d<uint, access::read> sym_texture [[texture(0)]];

// Decode
uint slot = state & (PROB_SCALE - 1);
uint8_t s = sym_texture.read(uint2(slot & 127, slot >> 7)).r;  // Texture read
```
