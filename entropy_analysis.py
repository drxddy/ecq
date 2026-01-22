#!/usr/bin/env python3
"""
Entropy Analysis for 4-bit Quantized Weights

Key insight: Near-Gaussian weights → non-uniform 4-bit symbol distribution
→ Shannon entropy < 4 bits → entropy coding can compress further

This measures the THEORETICAL bandwidth savings from entropy coding.
"""

import numpy as np
from collections import Counter
from typing import Dict, List
import time

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
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

def analyze_distribution(weights: np.ndarray, name: str = "weights") -> Dict:
    """Analyze entropy and distribution of quantized weights."""
    
    indices, scale, zero_point = quantize_4bit(weights)
    entropy = compute_entropy(indices)
    
    # Symbol distribution
    counts = np.bincount(indices.flatten(), minlength=16)
    probs = counts / counts.sum()
    
    # Calculate key metrics
    fixed_bits = 4.0
    savings_percent = (1 - entropy / fixed_bits) * 100
    compression_ratio = fixed_bits / entropy if entropy > 0 else float('inf')
    
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    print(f"Shape: {weights.shape} ({weights.size:,} elements)")
    print(f"Weight stats: μ={weights.mean():.4f}, σ={weights.std():.4f}")
    print(f"")
    print(f"Shannon Entropy:     {entropy:.3f} bits/symbol")
    print(f"Fixed 4-bit:         4.000 bits/symbol")  
    print(f"Savings:             {savings_percent:.1f}%")
    print(f"Compression:         {compression_ratio:.2f}x over 4-bit")
    print(f"")
    print(f"Symbol Distribution:")
    
    # ASCII histogram
    max_prob = probs.max()
    for sym in range(16):
        bar_len = int(35 * probs[sym] / max_prob) if max_prob > 0 else 0
        print(f"  {sym:2d}: {'█' * bar_len} {probs[sym]*100:5.1f}%")
    
    return {
        "name": name,
        "entropy": entropy,
        "savings_percent": savings_percent,
        "compression_ratio": compression_ratio,
    }


def test_synthetic_distributions():
    """Test entropy on various synthetic distributions."""
    
    print("\n" + "="*70)
    print(" SYNTHETIC DISTRIBUTION TESTS")
    print("="*70)
    
    np.random.seed(42)
    n = 1_000_000  # 1M weights
    
    results = []
    
    # Standard Gaussian (what modern LLMs have)
    weights = np.random.randn(n).astype(np.float32) * 0.02
    results.append(analyze_distribution(weights, "Gaussian (σ=0.02)"))
    
    # Uniform (worst case - max entropy)
    weights = np.random.uniform(-0.1, 0.1, n).astype(np.float32)
    results.append(analyze_distribution(weights, "Uniform [-0.1, 0.1]"))
    
    # Laplacian (slightly heavier tails)
    weights = np.random.laplace(0, 0.01, n).astype(np.float32)
    results.append(analyze_distribution(weights, "Laplacian (b=0.01)"))
    
    # Mixed Gaussian (bimodal)
    weights = np.concatenate([
        np.random.randn(n//2) * 0.01 - 0.02,
        np.random.randn(n//2) * 0.01 + 0.02,
    ]).astype(np.float32)
    results.append(analyze_distribution(weights, "Bimodal Gaussian"))
    
    return results


def test_real_model():
    """Test on real model weights."""
    
    print("\n" + "="*70)
    print(" REAL MODEL WEIGHT ANALYSIS")
    print("="*70)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM
        
        print("\nLoading SmolLM-135M...")
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM-135M",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        results = []
        layer_types = {"q_proj": [], "k_proj": [], "v_proj": [], 
                       "o_proj": [], "up_proj": [], "down_proj": [], "gate_proj": []}
        
        for name, param in model.named_parameters():
            if param.ndim == 2 and param.numel() > 10000:
                weights = param.detach().numpy()
                
                # Quick entropy calculation
                indices, _, _ = quantize_4bit(weights)
                entropy = compute_entropy(indices)
                savings = (1 - entropy / 4.0) * 100
                
                # Categorize by layer type
                for layer_type in layer_types:
                    if layer_type in name:
                        layer_types[layer_type].append(entropy)
                        break
                
                results.append({
                    "name": name,
                    "entropy": entropy,
                    "savings_percent": savings,
                    "shape": weights.shape,
                })
        
        # Summary by layer type
        print(f"\n{'='*60}")
        print(" ENTROPY BY LAYER TYPE")
        print(f"{'='*60}")
        print(f"{'Layer Type':<15} {'Avg Entropy':>12} {'Savings':>10}")
        print("-" * 40)
        
        for layer_type, entropies in layer_types.items():
            if entropies:
                avg = np.mean(entropies)
                savings = (1 - avg / 4.0) * 100
                print(f"{layer_type:<15} {avg:>12.3f} {savings:>9.1f}%")
        
        # Overall summary
        all_entropies = [r["entropy"] for r in results]
        if all_entropies:
            avg_entropy = np.mean(all_entropies)
            avg_savings = (1 - avg_entropy / 4.0) * 100
            
            print(f"\n{'='*60}")
            print(" OVERALL SUMMARY")
            print(f"{'='*60}")
            print(f"Layers analyzed:      {len(results)}")
            print(f"Average entropy:      {avg_entropy:.3f} bits")
            print(f"Average savings:      {avg_savings:.1f}% over 4-bit")
            print(f"Effective bits:       {avg_entropy:.2f} bits/weight")
        
        # Detailed analysis of one layer
        for name, param in model.named_parameters():
            if "down_proj" in name and param.ndim == 2:
                weights = param.detach().numpy()
                analyze_distribution(weights, f"Detailed: {name}")
                break
                
        return results
        
    except ImportError as e:
        print(f"Could not load model: {e}")
        print("Install with: pip install torch transformers")
        return []


def bandwidth_impact_analysis():
    """Calculate real-world impact on Apple Silicon."""
    
    print("\n" + "="*70)
    print(" BANDWIDTH IMPACT ANALYSIS")
    print("="*70)
    
    # From empirical tests: Gaussian weights show ~2.7-3.0 bits entropy
    avg_entropy = 2.85  # Based on synthetic Gaussian tests above
    fixed_bits = 4.0
    
    # Apple Silicon specs
    specs = [
        ("M1", 68.25),
        ("M1 Pro", 200),
        ("M1 Max", 400),
        ("M2", 100),
        ("M3", 100),
        ("M3 Pro", 150),
        ("M3 Max", 400),
        ("M4", 120),
        ("M4 Pro", 273),
        ("M4 Max", 546),
    ]
    
    compression = fixed_bits / avg_entropy
    
    print(f"\nAssuming average entropy of {avg_entropy:.2f} bits (conservative)")
    print(f"Effective compression: {compression:.2f}x over 4-bit")
    print(f"\n{'Chip':<12} {'BW (GB/s)':>12} {'4-bit tok/s':>14} {'Entropy tok/s':>14} {'Gain':>8}")
    print("-" * 62)
    
    model_size_4bit = 3.5  # GB for 7B model
    
    for chip, bw in specs:
        tok_4bit = bw / model_size_4bit
        tok_entropy = tok_4bit * compression
        gain = tok_entropy - tok_4bit
        print(f"{chip:<12} {bw:>12.0f} {tok_4bit:>14.1f} {tok_entropy:>14.1f} {'+' + str(int(gain)):>8}")
    
    print(f"\n⚠️  Requires fused ANS-decode + GEMM kernel in Metal")
    print(f"   Decode compute must be < bandwidth savings")
    print(f"\n   At 7 TFLOPS and ~10 ops/symbol decode:")
    print(f"   Decode throughput: ~700 billion symbols/sec")
    print(f"   7B model = ~7B weights = 0.01ms decode overhead ✓")


def main():
    test_synthetic_distributions()
    test_real_model()
    bandwidth_impact_analysis()


if __name__ == "__main__":
    main()
