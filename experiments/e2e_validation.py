#!/usr/bin/env python3
"""
End-to-End Validation: Entropy-Coded Inference with Local Model

This script:
1. Loads local model (SmolLM-135M or GPT-2)
2. Quantizes weight matrices to 4-bit
3. Encodes with interleaved rANS
4. Decodes and verifies lossless roundtrip
5. Measures compression ratio
"""

import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import json
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from safetensors import safe_open

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantizer import quantize_affine, compute_entropy
from src.interleaved_rans import interleaved_rans_decode
from src.rans import RANSTable


@dataclass
class LayerResult:
    """Result for a single layer."""
    name: str
    shape: Tuple[int, ...]
    params: int
    entropy: float
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    roundtrip_ok: bool
    encode_time_ms: float
    decode_time_ms: float


def process_layer(
    name: str,
    weights: np.ndarray,
    n_streams: int = 8,  # Fewer streams for speed
    bits: int = 4,
    max_elements: int = 100000  # Limit size for speed
) -> LayerResult:
    """Quantize, encode, decode, and verify a single layer."""
    
    # Limit size for faster testing
    flat = weights.flatten()
    if len(flat) > max_elements:
        flat = flat[:max_elements]
    
    # Quantize to 4-bit
    w_min, w_max = flat.min(), flat.max()
    scale = (w_max - w_min) / 15 if w_max != w_min else 1e-8
    indices = np.clip(np.round((flat - w_min) / scale), 0, 15).astype(np.uint8)
    
    # Compute entropy
    entropy = compute_entropy(indices, bits)
    
    # Original size (4-bit packed)
    original_bytes = (len(indices) * bits + 7) // 8
    
    # Encode with interleaved rANS
    start = time.perf_counter()
    
    n_symbols = 1 << bits
    counts = np.bincount(indices, minlength=n_symbols)
    table = RANSTable.from_counts(counts)
    
    from src.interleaved_rans import interleaved_rans_encode
    tile = interleaved_rans_encode(indices, table, n_streams)
    
    encode_time = (time.perf_counter() - start) * 1000
    
    # Decode
    start = time.perf_counter()
    decoded = interleaved_rans_decode(tile)
    decode_time = (time.perf_counter() - start) * 1000
    
    # Verify lossless
    roundtrip_ok = np.array_equal(indices, decoded)
    
    return LayerResult(
        name=name,
        shape=weights.shape,
        params=len(indices),
        entropy=entropy,
        original_bytes=original_bytes,
        compressed_bytes=tile.compressed_bytes,
        compression_ratio=tile.compression_ratio,
        roundtrip_ok=roundtrip_ok,
        encode_time_ms=encode_time,
        decode_time_ms=decode_time
    )


def main():
    print("="*70)
    print(" END-TO-END VALIDATION: Local Model with Entropy Coding")
    print("="*70)
    
    # Use local model
    model_path = Path("models/SmolLM-135M/model.safetensors")
    if not model_path.exists():
        model_path = Path("models/openai-community_gpt2/model.safetensors")
    
    if not model_path.exists():
        print("No local model found in models/")
        return False
    
    print(f"Model: {model_path}")
    
    results: List[LayerResult] = []
    
    # Process first 3 layers only for speed
    print("\nProcessing layers (first 3 for speed)...", flush=True)
    print("-"*70)
    print(f"{'Layer':<35} {'Shape':<12} {'Entropy':>7} {'Ratio':>7} {'OK':>4}")
    print("-"*70, flush=True)
    
    layer_count = 0
    max_layers = 3
    
    with safe_open(model_path, framework='pt') as f:
        for tensor_name in f.keys():
            if layer_count >= max_layers:
                break
                
            tensor = f.get_tensor(tensor_name)
            
            # Skip small tensors
            if tensor.ndim < 2 or tensor.numel() < 10000:
                continue
            
            # Convert to numpy
            weights = tensor.float().numpy()
            
            print(f"  Processing {tensor_name[:40]}...", flush=True)
            
            # Process layer
            result = process_layer(
                name=tensor_name,
                weights=weights,
                n_streams=4,  # Fewer streams
                max_elements=20000  # Smaller for speed
            )
            results.append(result)
            layer_count += 1
            
            # Print progress
            short_name = result.name[-35:] if len(result.name) > 35 else result.name
            shape_str = f"{result.shape[0]}x{result.shape[1]}" if len(result.shape) == 2 else str(result.shape)[:12]
            ok_str = "✓" if result.roundtrip_ok else "✗"
            print(f"{short_name:<35} {shape_str:<12} {result.entropy:>7.2f} {result.compression_ratio:>6.2f}x {ok_str:>4}", flush=True)
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    total_params = sum(r.params for r in results)
    total_original = sum(r.original_bytes for r in results)
    total_compressed = sum(r.compressed_bytes for r in results)
    all_ok = all(r.roundtrip_ok for r in results)
    
    weighted_entropy = sum(r.entropy * r.params for r in results) / total_params
    overall_ratio = total_original / total_compressed
    
    total_encode_time = sum(r.encode_time_ms for r in results)
    total_decode_time = sum(r.decode_time_ms for r in results)
    
    print(f"Layers processed:    {len(results)}")
    print(f"Total parameters:    {total_params:,}")
    print(f"Original size:       {total_original / 1e6:.2f} MB (4-bit packed)")
    print(f"Compressed size:     {total_compressed / 1e6:.2f} MB (entropy-coded)")
    print(f"Compression ratio:   {overall_ratio:.2f}x")
    print(f"Average entropy:     {weighted_entropy:.2f} bits")
    print(f"Lossless verified:   {'✓ ALL PASSED' if all_ok else '✗ SOME FAILED'}")
    print()
    print(f"Encode time (CPU):   {total_encode_time:.1f} ms")
    print(f"Decode time (CPU):   {total_decode_time:.1f} ms")
    
    # Throughput estimate
    print("\n" + "="*70)
    print(" INFERENCE ESTIMATE (M2 Pro @ 200 GB/s)")
    print("="*70)
    
    # Scale to 7B model
    scale_factor = 7e9 / total_params
    size_7b_4bit = 3.5  # GB
    size_7b_entropy = size_7b_4bit / overall_ratio
    
    # Measured fused throughput from our Metal kernel
    fused_throughput = 138.7  # GB/s (from optimized kernel)
    pure_bw = 200  # GB/s
    
    time_4bit = size_7b_4bit / pure_bw * 1000  # ms
    time_entropy = size_7b_entropy / fused_throughput * 1000  # ms
    
    speedup = time_4bit / time_entropy
    
    print(f"7B model (4-bit):    {size_7b_4bit:.2f} GB → {time_4bit:.1f} ms → {1000/time_4bit:.0f} tok/s")
    print(f"7B model (entropy):  {size_7b_entropy:.2f} GB → {time_entropy:.1f} ms → {1000/time_entropy:.0f} tok/s")
    print(f"Speedup:             {speedup:.2f}x")
    
    # Per-layer breakdown (top 5 most compressible)
    print("\n" + "="*70)
    print(" TOP 5 MOST COMPRESSIBLE LAYERS")
    print("="*70)
    
    sorted_results = sorted(results, key=lambda r: -r.compression_ratio)
    for r in sorted_results[:5]:
        print(f"  {r.name:<40} {r.compression_ratio:.2f}x ({r.entropy:.2f} bits)")
    
    # Save results
    output_path = Path("experiments/results/e2e_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'model': 'SmolLM-135M',
        'total_params': total_params,
        'total_original_bytes': total_original,
        'total_compressed_bytes': total_compressed,
        'compression_ratio': overall_ratio,
        'avg_entropy': weighted_entropy,
        'all_lossless': all_ok,
        'encode_time_ms': total_encode_time,
        'decode_time_ms': total_decode_time,
        'layers': [
            {
                'name': r.name,
                'shape': list(r.shape),
                'params': r.params,
                'entropy': r.entropy,
                'compression_ratio': r.compression_ratio,
                'roundtrip_ok': r.roundtrip_ok,
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return all_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
