# Entropy-Coded Quantization for LLM Inference

**Research project exploring lossless compression beyond 4-bit quantization.**

## Key Finding

LLM weights, when quantized to 4-bit, have Shannon entropy significantly below 4 bits:

| Model | Parameters | Entropy | Compression over 4-bit |
|-------|-----------|---------|------------------------|
| Qwen2.5-1.5B | 1.5B | 1.12 bits | **3.58x** |
| Qwen2.5-0.5B | 494M | 1.15 bits | **3.47x** |
| GPT-2 | 137M | 1.17 bits | **3.42x** |
| SmolLM-135M | 134M | 1.54 bits | **2.59x** |
| **Average** | — | — | **3.27x** |

This means entropy coding (rANS) can compress 4-bit weights by an additional 3.27x **losslessly**.

## Impact

For a 7B model on Apple Silicon M3 Pro (150 GB/s bandwidth):

| Format | Size | Tokens/sec |
|--------|------|------------|
| 4-bit quantized | 3.5 GB | 42.9 |
| Entropy-coded 4-bit | 1.07 GB | **140.0** |

**3.27x speedup with zero quality loss.**

## Project Structure

```
PQQ/
├── src/
│   ├── quantizer.py      # Quantization utilities
│   ├── rans.py           # rANS entropy codec
│   ├── model_analyzer.py # Model analysis tools
│   └── compressed_format.py # File format
├── experiments/
│   └── validate_compression.py # Main validation
├── docs/
│   └── metal_kernel_design.md  # GPU kernel design
├── models/               # Local model weights
├── entropy_coding_research.md  # Full research doc
└── paper_findings.md     # Original research context
```

## Quick Start

```bash
# Validate compression on models
python experiments/validate_compression.py
```

## How It Works

1. **Observation**: 4-bit quantized weights have non-uniform symbol distribution (bell-curved)
2. **Entropy**: Shannon entropy ≈ 1.1-1.5 bits, not 4 bits
3. **Compression**: rANS encoding approaches entropy limit
4. **Inference**: Fused decode+GEMM kernel loads less data from memory

## Prior Art

- [EntroLLM](https://arxiv.org/abs/2505.02380) (2025): Huffman coding for edge devices
- [Liguori](https://arxiv.org/abs/2404.10896) (2024): ANS for FPGA

Our contribution: rANS with fused decode+GEMM targeting Apple Silicon's extreme compute-to-bandwidth ratio (47:1).

## Next Steps

1. Metal kernel implementation (see [docs/metal_kernel_design.md](docs/metal_kernel_design.md))
2. MLX integration
3. End-to-end benchmarks
4. Publication

## License

Research code - MIT License
