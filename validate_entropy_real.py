#!/usr/bin/env python3
"""
Validate entropy coding potential on REAL model weights.
Uses local GPT-2 model from models/ directory.
"""

import numpy as np
from safetensors import safe_open
from pathlib import Path
from collections import Counter
from typing import Dict, List
import json

MODELS = [
    ("GPT-2", Path("models/openai-community_gpt2/model.safetensors")),
    ("SmolLM-135M", Path("models/SmolLM-135M/model.safetensors")),
]

def quantize_4bit(weights: np.ndarray) -> tuple:
    """Standard 4-bit affine quantization."""
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / 15
    if scale == 0:
        scale = 1e-8
    indices = np.clip(np.round((weights - w_min) / scale), 0, 15).astype(np.uint8)
    return indices, scale, w_min

def compute_entropy(indices: np.ndarray) -> float:
    """Compute Shannon entropy in bits per symbol."""
    counts = np.bincount(indices.flatten(), minlength=16)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def compute_kurtosis(weights: np.ndarray) -> float:
    """Compute excess kurtosis (Gaussian = 0)."""
    mean = weights.mean()
    std = weights.std()
    if std == 0:
        return 0
    return ((weights - mean) ** 4).mean() / (std ** 4) - 3

def analyze_layer(name: str, weights: np.ndarray) -> Dict:
    """Analyze a single layer's entropy characteristics."""
    indices, scale, zero_point = quantize_4bit(weights)
    entropy = compute_entropy(indices)
    kurtosis = compute_kurtosis(weights)
    
    savings_percent = (1 - entropy / 4.0) * 100
    compression = 4.0 / entropy if entropy > 0 else float('inf')
    
    # Symbol distribution
    counts = np.bincount(indices.flatten(), minlength=16)
    probs = counts / counts.sum()
    
    return {
        "name": name,
        "shape": weights.shape,
        "numel": weights.size,
        "entropy": entropy,
        "savings_percent": savings_percent,
        "compression": compression,
        "kurtosis": kurtosis,
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "probs": probs.tolist(),
    }


def analyze_model(model_name: str, model_path: Path):
    """Analyze a single model's entropy characteristics."""
    print("\n" + "="*70)
    print(f" ENTROPY ANALYSIS: {model_name}")
    print("="*70)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return None
    
    print(f"Loading {model_path}...")
    results = []
    total_params = 0
    weighted_entropy = 0
    
    with safe_open(model_path, framework="numpy") as f:
        tensor_names = list(f.keys())
        print(f"Found {len(tensor_names)} tensors\n")
        
        # Analyze each tensor
        for name in tensor_names:
            weights = f.get_tensor(name)
            
            # Skip 1D tensors (biases, layer norms)
            if weights.ndim < 2:
                continue
            
            result = analyze_layer(name, weights)
            results.append(result)
            
            total_params += result["numel"]
            weighted_entropy += result["entropy"] * result["numel"]
    
    # Sort by size (largest first)
    results.sort(key=lambda x: x["numel"], reverse=True)
    
    # Print detailed results
    print(f"{'Layer':<45} {'Shape':>15} {'Entropy':>8} {'Savings':>8} {'Kurt':>6}")
    print("-" * 85)
    
    for r in results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}" if len(r['shape']) == 2 else str(r['shape'])
        print(f"{r['name'][:44]:<45} {shape_str:>15} {r['entropy']:>8.3f} {r['savings_percent']:>7.1f}% {r['kurtosis']:>6.1f}")
    
    # Summary statistics
    avg_entropy = weighted_entropy / total_params
    avg_savings = (1 - avg_entropy / 4.0) * 100
    avg_compression = 4.0 / avg_entropy
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"Total weight matrices:  {len(results)}")
    print(f"Total parameters:       {total_params:,}")
    print(f"")
    print(f"Weighted avg entropy:   {avg_entropy:.3f} bits")
    print(f"Weighted avg savings:   {avg_savings:.1f}%")
    print(f"Effective compression:  {avg_compression:.2f}x over 4-bit")
    
    # Distribution analysis
    entropies = [r["entropy"] for r in results]
    print(f"\nEntropy range:          {min(entropies):.3f} - {max(entropies):.3f} bits")
    print(f"Entropy std:            {np.std(entropies):.3f}")
    
    # Bandwidth impact
    print("\n" + "="*70)
    print(" BANDWIDTH IMPACT (Apple Silicon)")
    print("="*70)
    
    model_size_fp16 = total_params * 2 / 1e9  # GB
    model_size_4bit = total_params * 0.5 / 1e9  # GB
    model_size_entropy = total_params * (avg_entropy / 8) / 1e9  # GB
    
    print(f"Model size (FP16):      {model_size_fp16:.2f} GB")
    print(f"Model size (4-bit):     {model_size_4bit:.2f} GB")
    print(f"Model size (entropy):   {model_size_entropy:.2f} GB")
    print(f"")
    
    bandwidth = 150  # M3 Pro GB/s
    tok_4bit = bandwidth / model_size_4bit if model_size_4bit > 0 else 0
    tok_entropy = bandwidth / model_size_entropy if model_size_entropy > 0 else 0
    
    print(f"At 150 GB/s (M3 Pro):")
    print(f"  4-bit:        {tok_4bit:.1f} tok/s")
    print(f"  Entropy-coded: {tok_entropy:.1f} tok/s")
    print(f"  Speedup:      {tok_entropy/tok_4bit:.2f}x")
    
    return {
        "model": model_name,
        "total_params": total_params,
        "avg_entropy": avg_entropy,
        "avg_savings": avg_savings,
        "compression": avg_compression,
    }


def main():
    """Test all available models."""
    print("="*70)
    print(" ENTROPY CODING VALIDATION ON LOCAL MODELS")
    print("="*70)
    
    summaries = []
    for name, path in MODELS:
        result = analyze_model(name, path)
        if result:
            summaries.append(result)
    
    # Final summary
    if summaries:
        print("\n" + "="*70)
        print(" FINAL SUMMARY - ALL MODELS")
        print("="*70)
        print(f"{'Model':<20} {'Params':>12} {'Entropy':>10} {'Savings':>10} {'Compression':>12}")
        print("-" * 66)
        for s in summaries:
            print(f"{s['model']:<20} {s['total_params']:>12,} {s['avg_entropy']:>10.3f} {s['avg_savings']:>9.1f}% {s['compression']:>11.2f}x")


if __name__ == "__main__":
    main()
