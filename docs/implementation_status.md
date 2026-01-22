# Implementation Status: Plan vs Actual

**Generated:** 2026-01-22  
**Repo:** ~/Developer/mlx-entropy (local changes vs origin/main)

---

## Executive Summary

| Category | Planned | Implemented | Status |
|----------|---------|-------------|--------|
| Python Layer | `EntropyCodedLinear` | ✅ Done | Complete |
| Decode Modes | FUSED, CACHED, GPU_ASYNC | ✅ FUSED, FUSED_V2, CACHED | 3/4 done |
| C++ Primitive | `EntropyCodedMatmul` | ✅ V1 + V2 | Complete |
| Metal Kernel | Fused decode+GEMV | ✅ V1 + V2 optimized | Complete |
| Python Bindings | `mx.entropy_coded_matmul` | ✅ V1 + V2 | Complete |
| Texture-backed tables | Planned | ❌ Not started | Pending |
| GPU_ASYNC mode | Planned | ❌ Not started | Pending |

---

## Phase-by-Phase Comparison

### Phase 1: Offline Encoder (Python-only) ✅ COMPLETE

| Planned | Actual Implementation |
|---------|----------------------|
| `python/mlx/nn/entropy_codec.py` | Integrated into `entropy_coded.py` |
| `EntropyCodedLinear` layer | ✅ `python/mlx/nn/layers/entropy_coded.py` |
| `entropy_quantize()` function | ✅ Implemented |
| rANS encoder/decoder | ✅ `entropy_encode()`, `entropy_encode_v2()` |

**Additional features implemented:**
- `RANSTable` class for frequency table management
- Per-tensor and per-group quantization support
- `DecodeMode` enum (FUSED, FUSED_V2, CACHED)

---

### Phase 2: Metal Decode Kernel ✅ COMPLETE

| Planned | Actual Implementation |
|---------|----------------------|
| `entropy_coded.metal` | ✅ `mlx/backend/metal/kernels/entropy_coded.metal` |
| `entropy_coded.h` | ✅ `mlx/backend/metal/kernels/entropy_coded.h` (15KB) |
| Parallel rANS decode | ✅ 64-256 interleaved streams |
| Shared memory freq tables | ✅ Threadgroup-cached |

**Metal kernel features:**
```
entropy_coded.h (V1):
├── Register-cached frequency tables (16 entries)
├── Threadgroup-cached symbol table (16KB)
├── Physical interleaving for coalesced access
├── SIMD reduction for output
└── Fused decode + dequantize + MAC
```

---

### Phase 3: Fused Decode+GEMV Kernel ✅ COMPLETE + OPTIMIZED

| Planned | Actual (V1) | Actual (V2) |
|---------|-------------|-------------|
| Fused decode+dequant+GEMV | ✅ `entropy_coded_qmv` | ✅ `entropy_coded_qmv_v2` |
| Decode work | O(rows × n) | O(n) ✅ **226x faster** |
| Per-row encoding | Not planned | ✅ Implemented |

**V2 Optimization (Not in original plan):**
```
entropy_coded_v2.h (10KB):
├── Per-row encoding → O(n) decode vs O(rows×n)
├── Row offsets + per-row stream lengths
├── Same optimizations as V1
└── 23-543x speedup over V1
```

**Benchmark Results:**
| Layer Size | V1→V2 Speedup | V2 vs CACHED Overhead |
|------------|---------------|----------------------|
| 256×512 | 23x | 0.9x (faster!) |
| 512×1024 | 113x | 1.5x |
| 1024×2048 | 543x | 1.5x |
| 2048×4096 | N/A | 1.1x |

---

### Phase 4: C++ Primitive Integration ✅ COMPLETE

| Planned | Actual |
|---------|--------|
| `EntropyCodedMatmul` class | ✅ `mlx/primitives.h` |
| `entropy_coded_matmul()` op | ✅ `mlx/ops.h`, `mlx/ops.cpp` |
| GPU dispatch | ✅ `mlx/backend/metal/quantized.cpp` |
| CPU fallback | ✅ `mlx/backend/cpu/quantized.cpp` |

**Additional (not in plan):**
- `EntropyCodedMatmulV2` class for per-row kernel
- `entropy_coded_matmul_v2()` operation

---

### Phase 5: Python Bindings ✅ COMPLETE

| Planned | Actual |
|---------|--------|
| `mx.entropy_coded_matmul` | ✅ `python/src/ops.cpp` |
| Layer export | ✅ `python/mlx/nn/layers/__init__.py` |

**Exports:**
```python
from mlx.nn.layers.entropy_coded import (
    EntropyCodedLinear,
    entropy_quantize,
    DecodeMode,
)
```

---

## Decode Strategies Comparison

From `docs/decode_strategies.md`:

| Strategy | Planned | Implemented | Notes |
|----------|---------|-------------|-------|
| FUSED (decode in kernel) | ✅ | ✅ V1 + V2 | V2 is 226x faster |
| CACHED (decode at load) | ✅ | ✅ | Works, fastest inference |
| GPU_ASYNC (Metal async) | ✅ | ❌ | Not started |
| Decode-once at load | ✅ | ✅ | Same as CACHED |
| LRU cache | ✅ | ❌ | Not implemented |
| Block-ANS codec | ✅ | ❌ | Not implemented |

---

## Hybrid Approach Analysis

From `docs/hybrid_approach_analysis.md`:

| Optimization | Planned | Implemented | Impact |
|--------------|---------|-------------|--------|
| Texture-backed sym_table | ✅ Priority 1 | ❌ | Est. 15% overhead reduction |
| Wider interleaving | ✅ Priority 2 | ✅ | 64-256 streams supported |
| Block structure | ✅ Priority 3 | ❌ | Better GPU utilization |
| Per-row encoding | Not planned | ✅ | **Biggest win: 226x speedup** |

**Unexpected optimization:** Per-row encoding (V2) was not in the original plan but provided the largest performance improvement by reducing decode work from O(rows × n) to O(n).

---

## Files Changed (git diff origin/main)

```
 mlx/backend/cpu/quantized.cpp             |  16 +++
 mlx/backend/metal/kernels/CMakeLists.txt  |   1 +
 mlx/backend/metal/quantized.cpp           | 121 +++++++++++++++
 mlx/ops.cpp                               | 124 ++++++++++++++++
 mlx/ops.h                                 |  34 +++++
 mlx/primitives.cpp                        |  35 ++++-
 mlx/primitives.h                          |  71 +++++++++
 python/mlx/nn/layers/__init__.py          |   5 +
 python/src/ops.cpp                        |  89 ++++++++++++
 
 New files (not in diff):
 mlx/backend/metal/kernels/entropy_coded.h       (15KB)
 mlx/backend/metal/kernels/entropy_coded_v2.h    (10KB)
 mlx/backend/metal/kernels/entropy_coded.metal   (7KB)
 python/mlx/nn/layers/entropy_coded.py           (new)
 examples/*.py                                   (12 files)
```

---

## Performance Summary

| Metric | Planned | Achieved |
|--------|---------|----------|
| Compression ratio | 1.84x over 4-bit | ✅ 1.84x |
| V1 FUSED overhead | 24-29% | ❌ 20-600x slower than CACHED |
| V2 FUSED overhead | N/A (not planned) | ✅ 1.1-1.5x vs CACHED |
| Effective speedup | 2.27x | ✅ ~2x with V2 FUSED |

---

## Remaining Work

### High Priority
1. ~~**GPU_ASYNC mode** - Decode next layer while computing current~~ ✅ **IMPLEMENTED**
2. **Texture-backed sym_table** - Could reduce remaining 1.1-1.5x overhead

### Medium Priority
3. **Block-structured encoding** - Better parallelism for very large layers
4. **LRU cache** - Memory-adaptive decoded weight caching

### Low Priority
5. **ASTC hybrid** - Use ASTC for high-entropy layers
6. **mlx-lm integration** - Model converter and loader

---

## GPU_ASYNC Implementation (New)

**Status:** ✅ Implemented

The GPU_ASYNC mode decodes the next layer's weights while the current layer is computing,
using a separate GPU stream for overlap.

### Components Added:
- `entropy_decode_async.h` - Metal decode-only kernel
- `EntropyDecodeAsync` primitive in C++
- `mx.entropy_decode_async()` Python binding
- `AsyncDecodePrefetcher` class with double-buffering
- `_forward_gpu_async()` method in EntropyCodedLinear

### Benchmark Results:
| Config | CACHED | FUSED_V2 | GPU_ASYNC |
|--------|--------|----------|-----------|
| 4-layer (512) | 0.51ms | 0.60ms | 0.63ms |
| 6-layer (256) | 0.63ms | 0.51ms | 0.55ms |

### Conclusion:
GPU_ASYNC mode works correctly but provides minimal benefit for small/medium models
because the V2 fused kernel is already very fast (1.1-1.5x overhead vs CACHED).
The async prefetch overhead exceeds the benefit for compute-bound workloads.

**Recommendation:** Use **FUSED_V2** mode for most use cases. Reserve GPU_ASYNC for
very deep models (50+ layers) where decode latency accumulates significantly.

---

## Conclusion

The implementation exceeds the original plan in several ways:

1. **V2 kernel optimization** (not planned) provides 226x speedup over V1
2. **FUSED_V2 mode** makes fused inference practical (1.1-1.5x overhead vs CACHED)
3. **Full C++ integration** with primitives, ops, and Python bindings
4. **Comprehensive benchmarking** with 12 example/test scripts

Main gaps:
- GPU_ASYNC mode not implemented
- Texture-backed tables not implemented
- No mlx-lm integration yet
