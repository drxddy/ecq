#!/usr/bin/env python3
"""
Experiment: Analyze entropy of quantized weights across multiple models.

This script measures the compression potential of entropy coding
for various LLM architectures and sizes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_analyzer import (
    analyze_safetensors,
    analyze_huggingface_model,
    print_model_stats,
    ModelStats,
)
import json
from datetime import datetime


# Models to analyze
LOCAL_MODELS = [
    ("GPT-2", Path("models/openai-community_gpt2/model.safetensors")),
    ("SmolLM-135M", Path("models/SmolLM-135M/model.safetensors")),
]

HUGGINGFACE_MODELS = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
]


def main():
    print("="*70)
    print(" Entropy Coding Compression Analysis")
    print(" " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    results = []
    
    # Analyze local models
    print("\n[1/2] Analyzing local models...")
    for name, path in LOCAL_MODELS:
        if path.exists():
            stats = analyze_safetensors(path, name)
            print_model_stats(stats)
            results.append(stats)
        else:
            print(f"  Skipping {name}: {path} not found")
    
    # Analyze HuggingFace models
    print("\n[2/2] Analyzing HuggingFace models...")
    for name, repo_id in HUGGINGFACE_MODELS:
        try:
            stats = analyze_huggingface_model(repo_id, name)
            print_model_stats(stats)
            results.append(stats)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
    
    # Summary table
    print("\n" + "="*70)
    print(" SUMMARY: Compression Potential")
    print("="*70)
    print(f"{'Model':<20} {'Params':>12} {'Entropy':>10} {'Savings':>10} {'Compression':>12}")
    print("-"*66)
    
    for stats in results:
        print(f"{stats.model_name:<20} "
              f"{stats.total_params:>12,} "
              f"{stats.weighted_entropy:>10.3f} "
              f"{stats.weighted_savings:>9.1f}% "
              f"{stats.weighted_compression:>11.2f}x")
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"entropy_analysis_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'models': [s.to_dict() for s in results],
            'summary': {
                'total_models': len(results),
                'avg_entropy': sum(s.weighted_entropy for s in results) / len(results),
                'avg_compression': sum(s.weighted_compression for s in results) / len(results),
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Key findings
    print("\n" + "="*70)
    print(" KEY FINDINGS")
    print("="*70)
    
    if results:
        best = max(results, key=lambda x: x.weighted_compression)
        worst = min(results, key=lambda x: x.weighted_compression)
        avg_comp = sum(s.weighted_compression for s in results) / len(results)
        
        print(f"Best compression:   {best.model_name} ({best.weighted_compression:.2f}x)")
        print(f"Worst compression:  {worst.model_name} ({worst.weighted_compression:.2f}x)")
        print(f"Average compression: {avg_comp:.2f}x over 4-bit quantization")
        print()
        print("Conclusion: Entropy coding provides significant lossless compression")
        print(f"            beyond fixed-width 4-bit quantization ({avg_comp:.1f}x average).")


if __name__ == "__main__":
    main()
