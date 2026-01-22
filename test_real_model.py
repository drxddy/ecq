"""
Test Polynomial Quantization on Real Model Weights
===================================================
Loads a small LLM and validates the quantization approach
on actual weight distributions.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

# Import our validation functions
from validate_compression import (
    affine_quantize,
    polynomial_quantize_nearest,
    polynomial_quantize_mse_optimal,
    lut_quantize,
    compute_metrics,
    POLY_DEGREE,
    NUM_BINS,
)


def analyze_weight_distribution(weights: np.ndarray, name: str) -> dict:
    """Compute distribution statistics for a weight tensor."""
    return {
        "name": name,
        "shape": weights.shape,
        "size": weights.size,
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "kurtosis": float(compute_kurtosis(weights)),
        "skewness": float(compute_skewness(weights)),
    }


def compute_kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis (normal = 0, heavy-tailed > 0)."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return np.mean(((x - mean) / std) ** 4) - 3


def compute_skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return np.mean(((x - mean) / std) ** 3)


def quantize_and_compare(weights: np.ndarray) -> dict:
    """Run all quantization methods on a weight tensor."""
    # Flatten for quantization
    w_flat = weights.flatten().astype(np.float64)
    
    affine = affine_quantize(w_flat)
    poly = polynomial_quantize_nearest(w_flat)
    poly_mse = polynomial_quantize_mse_optimal(w_flat)
    lut = lut_quantize(w_flat)
    
    return {
        "affine": compute_metrics(w_flat, affine.reconstructed),
        "poly": compute_metrics(w_flat, poly.reconstructed),
        "poly_mse": compute_metrics(w_flat, poly_mse.reconstructed),
        "lut": compute_metrics(w_flat, lut.reconstructed),
        "poly_r2": poly.r_squared,
        "poly_mse_r2": poly_mse.r_squared,
        "kurtosis": compute_kurtosis(w_flat),
    }


def load_model_weights(model_path: str) -> dict:
    """Load model weights from safetensors, downloading from HF if needed."""
    weights = {}
    
    model_dir = Path(model_path)
    
    # If it's a HuggingFace model ID, download it first
    if not model_dir.exists() and "/" in model_path:
        print(f"Downloading model from HuggingFace: {model_path}")
        try:
            from huggingface_hub import snapshot_download
            model_dir = Path(snapshot_download(
                model_path, 
                allow_patterns=["*.safetensors", "config.json"],
                local_dir=f"./models/{model_path.replace('/', '_')}"
            ))
            print(f"Downloaded to: {model_dir}")
        except Exception as e:
            print(f"Download failed: {e}")
            return {}
    
    # Try safetensors first
    safetensor_files = list(model_dir.glob("*.safetensors"))
    if safetensor_files:
        try:
            from safetensors import safe_open
            import torch  # For bfloat16 support
            for sf_path in safetensor_files:
                print(f"Loading: {sf_path.name}")
                with safe_open(sf_path, framework="pt") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        # Convert to float32 numpy
                        weights[key] = tensor.float().numpy()
            return weights
        except ImportError:
            print("safetensors not installed")
            pass
    
    # Try NPZ weights
    weight_files = list(model_dir.glob("*.npz"))
    if weight_files:
        for wf in weight_files:
            data = dict(np.load(wf))
            weights.update(data)
        return weights
    
    return weights


def run_real_model_test(model_path: str, max_layers: int = None):
    """
    Test quantization on real model weights.
    """
    print("=" * 80)
    print(f"TESTING ON REAL MODEL: {model_path}")
    print("=" * 80)
    
    print("\nLoading model weights...")
    weights = load_model_weights(model_path)
    
    if not weights:
        print("Failed to load weights.")
        return
    
    print(f"Loaded {len(weights)} weight tensors")
    
    # Filter to only dense/linear layers (skip embeddings, norms)
    linear_weights = {
        k: v for k, v in weights.items() 
        if ('weight' in k.lower() or 'proj' in k.lower())
        and 'norm' not in k.lower()
        and 'embed' not in k.lower()
        and v.ndim >= 2
        and v.size >= 1024  # Skip tiny tensors
    }
    
    print(f"Found {len(linear_weights)} linear layer weights")
    
    if max_layers:
        keys = list(linear_weights.keys())[:max_layers]
        linear_weights = {k: linear_weights[k] for k in keys}
        print(f"Testing on first {max_layers} layers")
    
    # Collect results by layer type
    results_by_type = defaultdict(list)
    all_results = []
    
    print("\n" + "-" * 80)
    print(f"{'Layer':<45} {'Kurt':>6} {'Affine':>7} {'PolyMSE':>7} {'Delta':>6}")
    print("-" * 80)
    
    for name, w in linear_weights.items():
        w_np = np.array(w).astype(np.float64)
        
        # Quantize per-channel (row) for realistic simulation
        if w_np.ndim >= 2:
            # Sample a few channels to speed up
            n_channels = min(32, w_np.shape[0])
            channel_results = []
            
            for i in range(0, w_np.shape[0], max(1, w_np.shape[0] // n_channels)):
                channel = w_np[i].flatten()
                if len(channel) < 16:
                    continue
                res = quantize_and_compare(channel)
                channel_results.append(res)
            
            if not channel_results:
                continue
                
            # Average across channels
            avg_affine = np.mean([r["affine"]["sqnr_db"] for r in channel_results])
            avg_poly = np.mean([r["poly"]["sqnr_db"] for r in channel_results])
            avg_poly_mse = np.mean([r["poly_mse"]["sqnr_db"] for r in channel_results])
            avg_kurtosis = np.mean([r["kurtosis"] for r in channel_results])
            avg_r2 = np.mean([r["poly_r2"] for r in channel_results])
            
            delta = avg_poly - avg_affine
            delta_mse = avg_poly_mse - avg_affine
            
            # Determine layer type
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                layer_type = 'attention_qkv'
            elif 'o_proj' in name or 'out_proj' in name:
                layer_type = 'attention_out'
            elif 'gate' in name or 'up_proj' in name:
                layer_type = 'mlp_gate'
            elif 'down_proj' in name:
                layer_type = 'mlp_down'
            else:
                layer_type = 'other'
            
            result = {
                "name": name,
                "type": layer_type,
                "kurtosis": avg_kurtosis,
                "affine_sqnr": avg_affine,
                "poly_sqnr": avg_poly,
                "poly_mse_sqnr": avg_poly_mse,
                "delta": delta,
                "delta_mse": delta_mse,
                "r2": avg_r2,
            }
            all_results.append(result)
            results_by_type[layer_type].append(result)
            
            # Print row
            short_name = name[-48:] if len(name) > 48 else name
            marker = "✓" if delta_mse >= -0.01 else " "
            print(f"{short_name:<45} {avg_kurtosis:>6.1f} {avg_affine:>7.2f} {avg_poly_mse:>7.2f} {delta_mse:>+6.2f}{marker}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY LAYER TYPE")
    print("=" * 80)
    
    print(f"\n{'Layer Type':<20} {'Count':>6} {'Affine':>10} {'PolyMSE':>10} {'Delta':>10}")
    print("-" * 60)
    
    poly_wins = 0
    total = 0
    
    for layer_type, results in sorted(results_by_type.items()):
        n = len(results)
        avg_affine = np.mean([r["affine_sqnr"] for r in results])
        avg_poly_mse = np.mean([r["poly_mse_sqnr"] for r in results])
        avg_delta = np.mean([r["delta_mse"] for r in results])
        
        wins = sum(1 for r in results if r["delta_mse"] >= -0.01)
        poly_wins += wins
        total += n
        
        print(f"{layer_type:<20} {n:>6} {avg_affine:>9.2f}dB {avg_poly_mse:>9.2f}dB {avg_delta:>+9.2f}dB")
    
    print("-" * 60)
    
    # Overall
    if all_results:
        overall_affine = np.mean([r["affine_sqnr"] for r in all_results])
        overall_poly_mse = np.mean([r["poly_mse_sqnr"] for r in all_results])
        overall_delta = np.mean([r["delta_mse"] for r in all_results])
        overall_kurt = np.mean([r["kurtosis"] for r in all_results])
        
        print(f"\n{'OVERALL':<20} {len(all_results):>6} {overall_affine:>9.2f}dB {overall_poly_mse:>9.2f}dB {overall_delta:>+9.2f}dB")
        print(f"\nPolyMSE matches Affine on {poly_wins}/{total} layers ({100*poly_wins/total:.1f}%)")
        
        # Find layers where poly would win (high kurtosis)
        high_kurt_layers = [r for r in all_results if r["kurtosis"] > 3]
        if high_kurt_layers:
            hk_delta = np.mean([r["delta"] for r in high_kurt_layers])
            print(f"\nLayers with kurtosis > 3: {len(high_kurt_layers)}")
            print(f"  Average delta on high-kurtosis layers: {hk_delta:+.2f} dB")
        
        # Recommendation
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        print(f"""
  Model Weight Distribution:
  - Average Kurtosis: {overall_kurt:.2f}
  - Kurtosis > 3 (heavy-tailed): {sum(1 for r in all_results if r['kurtosis'] > 3)} layers
  - Kurtosis < 1 (light-tailed): {sum(1 for r in all_results if r['kurtosis'] < 1)} layers
  
  Result: Polynomial {overall_delta:+.2f} dB vs Affine
  
  INSIGHT:
  This model has {'LIGHT' if overall_kurt < 2 else 'HEAVY'}-tailed weight distributions.
  {'Affine quantization is optimal for light-tailed (near-Gaussian) distributions.' if overall_kurt < 2 else 'Polynomial quantization should help here.'}
  
  For the procedural quantization thesis to hold, we need models with:
  - Kurtosis > 3 (heavy tails from training dynamics)
  - This is common in larger models trained longer (7B+)
  
  SmolLM-135M appears to have near-Gaussian weights, likely due to:
  - Small model size (less extreme weight distributions)
  - Training regularization
  - Modern initialization schemes
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test quantization on real model")
    parser.add_argument("model", type=str, nargs="?", 
                        default="mlx-community/SmolLM-135M-Instruct-4bit",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Max layers to test (for speed)")
    
    args = parser.parse_args()
    run_real_model_test(args.model, args.max_layers)
