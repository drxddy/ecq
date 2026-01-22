#!/usr/bin/env python3
"""
Validate entropy-based compression potential across models.

This proves the research direction without slow Python rANS codec.
Shannon entropy gives us the theoretical minimum bits needed.
"""

import numpy as np
from pathlib import Path
from safetensors import safe_open
import torch
from huggingface_hub import hf_hub_download
from datetime import datetime
import json


def compute_entropy(indices: np.ndarray, bits: int = 4) -> float:
    """Shannon entropy in bits per symbol."""
    n_symbols = 1 << bits
    counts = np.bincount(indices.flatten(), minlength=n_symbols)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def analyze_model(name: str, path: Path = None, repo_id: str = None) -> dict:
    """Analyze a model's compression potential."""
    print(f"\n{'='*60}")
    print(f" {name}")
    print('='*60)
    
    # Get path
    if path and path.exists():
        model_path = path
    elif repo_id:
        try:
            model_path = Path(hf_hub_download(repo_id, 'model.safetensors'))
        except:
            model_path = Path(hf_hub_download(repo_id, 'model-00001-of-00002.safetensors'))
    else:
        print("  Model not found")
        return None
    
    total_params = 0
    weighted_entropy = 0
    layer_count = 0
    
    with safe_open(model_path, framework='pt') as f:
        for tensor_name in f.keys():
            w = f.get_tensor(tensor_name)
            
            if w.ndim < 2 or w.numel() < 10000:
                continue
            
            # Convert to float32 numpy
            w_np = w.float().numpy()
            
            # 4-bit quantization
            w_min, w_max = w_np.min(), w_np.max()
            scale = (w_max - w_min) / 15
            if scale == 0:
                continue
            indices = np.clip(np.round((w_np - w_min) / scale), 0, 15).astype(np.uint8)
            
            # Entropy
            entropy = compute_entropy(indices)
            
            total_params += w_np.size
            weighted_entropy += entropy * w_np.size
            layer_count += 1
    
    if total_params == 0:
        return None
    
    avg_entropy = weighted_entropy / total_params
    savings = (1 - avg_entropy / 4.0) * 100
    compression = 4.0 / avg_entropy
    
    print(f"Layers:      {layer_count}")
    print(f"Parameters:  {total_params:,}")
    print(f"Entropy:     {avg_entropy:.3f} bits")
    print(f"Savings:     {savings:.1f}%")
    print(f"Compression: {compression:.2f}x over 4-bit")
    
    return {
        'model': name,
        'params': total_params,
        'entropy': avg_entropy,
        'savings': savings,
        'compression': compression,
    }


def main():
    print("="*60)
    print(" ENTROPY CODING VALIDATION")
    print(" " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("="*60)
    
    results = []
    
    # Local models
    local = [
        ("GPT-2", Path("models/openai-community_gpt2/model.safetensors")),
        ("SmolLM-135M", Path("models/SmolLM-135M/model.safetensors")),
    ]
    
    for name, path in local:
        r = analyze_model(name, path=path)
        if r:
            results.append(r)
    
    # HuggingFace models
    hf = [
        ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
        ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ]
    
    for name, repo in hf:
        r = analyze_model(name, repo_id=repo)
        if r:
            results.append(r)
    
    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Params':>12} {'Entropy':>8} {'Compression':>12}")
    print("-"*54)
    
    for r in sorted(results, key=lambda x: -x['compression']):
        print(f"{r['model']:<20} {r['params']:>12,} {r['entropy']:>8.2f} {r['compression']:>11.2f}x")
    
    avg_comp = np.mean([r['compression'] for r in results])
    print("-"*54)
    print(f"{'AVERAGE':<20} {'':<12} {'':<8} {avg_comp:>11.2f}x")
    
    # Bandwidth impact
    print("\n" + "="*60)
    print(" BANDWIDTH IMPACT (7B model @ 150 GB/s)")
    print("="*60)
    
    model_7b_4bit = 3.5  # GB
    model_7b_entropy = model_7b_4bit / avg_comp
    
    tok_4bit = 150 / model_7b_4bit
    tok_entropy = 150 / model_7b_entropy
    
    print(f"4-bit size:        {model_7b_4bit:.2f} GB")
    print(f"Entropy-coded:     {model_7b_entropy:.2f} GB")
    print(f"4-bit tok/s:       {tok_4bit:.1f}")
    print(f"Entropy tok/s:     {tok_entropy:.1f}")
    print(f"Speedup:           {avg_comp:.2f}x")
    
    # Save
    output = Path("experiments/results")
    output.mkdir(parents=True, exist_ok=True)
    
    with open(output / "validation.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'avg_compression': avg_comp,
        }, f, indent=2)
    
    print(f"\nSaved to: {output / 'validation.json'}")


if __name__ == "__main__":
    main()
