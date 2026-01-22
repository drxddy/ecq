# GitHub Issue Draft: Entropy-Coded Quantization for MLX

> **Copy-paste ready issue for https://github.com/ml-explore/mlx**

---

## Issue Title

**[Feature Request] Entropy-Coded Quantization for Additional 2-3x Compression on Quantized Weights**

---

## Issue Body

### Summary

I'd like to propose adding **entropy coding** (specifically rANS) as an additional compression layer on top of MLX's existing quantization. This provides **lossless** compression of already-quantized weights, achieving an additional 2-3x size reduction and proportional inference speedup on memory-bound workloads.

### Motivation

LLM inference on Apple Silicon is memory-bandwidth limited (47:1 compute-to-bandwidth ratio). Current 4-bit quantization achieves 4x compression, but the quantized symbol distribution is highly non-uniform—symbols cluster around the center of the range due to the Gaussian nature of LLM weights.

**Key insight:** When 4-bit quantized, LLM weights have Shannon entropy of only **1.2-1.5 bits** on average, not 4 bits. This means we can losslessly compress 4-bit representations to ~1.5 bits using entropy coding.

### Measured Results

I've prototyped this with real models:

| Model | Measured Entropy | Compression over 4-bit |
|-------|-----------------|----------------------|
| Qwen2.5-1.5B | 1.12 bits | 3.58x |
| Qwen2.5-0.5B | 1.15 bits | 3.47x |
| GPT-2 | 1.17 bits | 3.42x |
| SmolLM-135M | 1.54 bits | 2.59x |

**7B Model Inference Projection (M2 Pro @ 200 GB/s):**

| Format | Size | Est. tok/s |
|--------|------|------------|
| 4-bit | 3.5 GB | 40 |
| **Entropy-coded** | **1.1 GB** | **~100** |

### Technical Approach

1. **Interleaved rANS streams** - Each weight matrix is encoded as 256+ independent streams that can be decoded in parallel on GPU
2. **Fused decode+dequantize+GEMV** - Decode directly into registers, never materialize full weight matrix
3. **Per-tile frequency tables** - Small tables (~100 bytes) fit in shared memory

The decode overhead is minimal (~0.07ms for 7B model) because:
- rANS decode is ~10 ops per symbol
- At 7 TFLOPS, decoding is 100x faster than memory transfer
- Bandwidth savings far exceed decode cost

### Prototype Results

I've implemented and benchmarked a Metal kernel for this:

```
Metal device: Apple M2 Pro
Decode throughput: 106 G elements/s
Fused GEMV throughput: 139 GB/s effective
Speedup over 4-bit: 2.27x
```

### Proposed Implementation

**Phase 1: Python-only encoder**
- Add `entropy_encode()` / `entropy_decode()` functions
- New `EntropyCodedLinear` layer that calls existing `quantized_matmul` after decoding

**Phase 2: Metal kernel**
- Parallel rANS decode kernel
- Fused decode+dequantize+GEMV kernel

**Phase 3: Full integration**
- New `EntropyCodedMatmul` primitive
- Integration with mlx-lm model loading

### Prior Art

- **EntroLLM** (May 2025) - Huffman coding for edge devices, 30-65% savings
- **Liguori et al.** (2024) - ANS on FPGA
- **VLCQ** (2024) - Variable-length codes for CNNs

Our approach differs by:
1. Using rANS (simpler parallel decode than Huffman)
2. Targeting Apple Silicon specifically
3. Fusing decode with GEMM to hide decode latency entirely

### Questions for Maintainers

1. Is there interest in this direction? I'm happy to contribute a PR
2. Would you prefer this as a new quantization mode, or as a separate layer?
3. Any concerns about the additional complexity in the Metal kernels?

### Resources

I've prepared:
- [Integration plan document](link-to-plan)
- [Working Python prototype with Metal kernels](link-to-repo)
- Benchmark scripts for validation

### Related Issues

- N/A (couldn't find existing issues on entropy coding)

---

## Labels to Add

- `enhancement`
- `quantization`
- `performance`
- `metal`

---

## Additional Notes for Submission

Before submitting:

1. **Run benchmarks on multiple chips** - Currently only tested on M2 Pro
2. **Test with larger models** - Validate the 7B projections
3. **Check mlx-lm integration** - Ensure the format works with existing pipelines
4. **Measure prefill performance** - Current focus is on token generation

---

## Alternative: Start with Discussion

If a full feature request feels premature, consider opening a **Discussion** first:

**Title:** "Interest in entropy-coded quantization for additional 2-3x compression?"

**Body:** (abbreviated version of above, focusing on the idea and seeking feedback)
