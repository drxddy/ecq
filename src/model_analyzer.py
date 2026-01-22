"""
Model weight analyzer for entropy coding research.

Analyzes LLM weights to measure compression potential from entropy coding.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
import json

from .quantizer import quantize_affine, compute_entropy, compute_compression_potential


@dataclass
class LayerStats:
    """Statistics for a single layer."""
    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: str
    
    # Weight statistics
    mean: float
    std: float
    min_val: float
    max_val: float
    kurtosis: float
    
    # Entropy statistics
    entropy: float
    savings_percent: float
    compression_ratio: float
    
    # Symbol distribution
    symbol_probs: List[float]


@dataclass
class ModelStats:
    """Aggregate statistics for a model."""
    model_name: str
    model_path: str
    total_params: int
    total_layers: int
    
    # Weighted averages
    weighted_entropy: float
    weighted_savings: float
    weighted_compression: float
    
    # Per-layer stats
    layers: List[LayerStats] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'total_params': self.total_params,
            'total_layers': self.total_layers,
            'weighted_entropy': self.weighted_entropy,
            'weighted_savings': self.weighted_savings,
            'weighted_compression': self.weighted_compression,
            'layers': [
                {
                    'name': l.name,
                    'shape': l.shape,
                    'numel': l.numel,
                    'entropy': l.entropy,
                    'savings_percent': l.savings_percent,
                    'compression_ratio': l.compression_ratio,
                    'kurtosis': l.kurtosis,
                }
                for l in self.layers
            ]
        }
    
    def save(self, path: Path):
        """Save stats to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def compute_kurtosis(weights: np.ndarray) -> float:
    """Compute excess kurtosis (Gaussian = 0)."""
    mean = weights.mean()
    std = weights.std()
    if std == 0:
        return 0.0
    return float(((weights - mean) ** 4).mean() / (std ** 4) - 3)


def analyze_layer(
    name: str,
    weights: np.ndarray,
    bits: int = 4,
    min_size: int = 1000
) -> Optional[LayerStats]:
    """
    Analyze a single layer's compression potential.
    
    Args:
        name: Layer name
        weights: Weight tensor
        bits: Quantization bit width
        min_size: Minimum tensor size to analyze
        
    Returns:
        LayerStats or None if tensor is too small
    """
    if weights.ndim < 2 or weights.size < min_size:
        return None
    
    # Convert to float32
    weights = weights.astype(np.float32)
    
    # Quantize
    quantized = quantize_affine(weights, bits=bits)
    
    # Compute entropy
    potential = compute_compression_potential(quantized.indices, bits)
    
    return LayerStats(
        name=name,
        shape=weights.shape,
        numel=weights.size,
        dtype=str(weights.dtype),
        mean=float(weights.mean()),
        std=float(weights.std()),
        min_val=float(weights.min()),
        max_val=float(weights.max()),
        kurtosis=compute_kurtosis(weights),
        entropy=potential['entropy'],
        savings_percent=potential['savings_percent'],
        compression_ratio=potential['compression_ratio'],
        symbol_probs=potential['symbol_distribution'],
    )


def analyze_safetensors(
    path: Path,
    model_name: str,
    bits: int = 4,
    min_size: int = 10000,
    use_torch: bool = True,
) -> ModelStats:
    """
    Analyze a safetensors model file.
    
    Args:
        path: Path to model.safetensors
        model_name: Human-readable model name
        bits: Quantization bit width
        min_size: Minimum tensor size to include
        use_torch: Use PyTorch for bfloat16 support
        
    Returns:
        ModelStats with per-layer analysis
    """
    from safetensors import safe_open
    
    framework = 'pt' if use_torch else 'numpy'
    
    layers = []
    total_params = 0
    weighted_entropy = 0.0
    
    with safe_open(path, framework=framework) as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            
            # Convert to numpy
            if use_torch:
                import torch
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.float().numpy()
            
            # Analyze layer
            stats = analyze_layer(name, tensor, bits, min_size)
            if stats is None:
                continue
            
            layers.append(stats)
            total_params += stats.numel
            weighted_entropy += stats.entropy * stats.numel
    
    # Compute weighted averages
    if total_params > 0:
        avg_entropy = weighted_entropy / total_params
        avg_savings = (1 - avg_entropy / bits) * 100
        avg_compression = bits / avg_entropy if avg_entropy > 0 else float('inf')
    else:
        avg_entropy = avg_savings = avg_compression = 0.0
    
    return ModelStats(
        model_name=model_name,
        model_path=str(path),
        total_params=total_params,
        total_layers=len(layers),
        weighted_entropy=avg_entropy,
        weighted_savings=avg_savings,
        weighted_compression=avg_compression,
        layers=layers,
    )


def analyze_huggingface_model(
    repo_id: str,
    model_name: Optional[str] = None,
    bits: int = 4,
    cache_dir: Optional[Path] = None,
) -> ModelStats:
    """
    Download and analyze a HuggingFace model.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., 'Qwen/Qwen2.5-0.5B')
        model_name: Human-readable name (defaults to repo_id)
        bits: Quantization bit width
        cache_dir: Optional cache directory
        
    Returns:
        ModelStats with analysis
    """
    from huggingface_hub import hf_hub_download
    
    if model_name is None:
        model_name = repo_id.split('/')[-1]
    
    # Try to download model.safetensors
    try:
        path = hf_hub_download(
            repo_id, 
            'model.safetensors',
            cache_dir=cache_dir
        )
    except Exception:
        # Try sharded model
        path = hf_hub_download(
            repo_id,
            'model-00001-of-00002.safetensors',
            cache_dir=cache_dir
        )
    
    return analyze_safetensors(Path(path), model_name, bits)


def print_model_stats(stats: ModelStats, detailed: bool = False):
    """Pretty-print model statistics."""
    print(f"\n{'='*70}")
    print(f" {stats.model_name}")
    print('='*70)
    print(f"Path:              {stats.model_path}")
    print(f"Total parameters:  {stats.total_params:,}")
    print(f"Layers analyzed:   {stats.total_layers}")
    print()
    print(f"Weighted entropy:  {stats.weighted_entropy:.3f} bits")
    print(f"Savings over 4-bit: {stats.weighted_savings:.1f}%")
    print(f"Compression ratio: {stats.weighted_compression:.2f}x")
    
    if detailed and stats.layers:
        print(f"\n{'Layer':<45} {'Shape':>15} {'Entropy':>8} {'Savings':>8}")
        print('-' * 80)
        
        # Sort by entropy (most compressible first)
        sorted_layers = sorted(stats.layers, key=lambda x: x.entropy)
        
        for layer in sorted_layers[:20]:  # Top 20
            shape_str = 'x'.join(map(str, layer.shape))
            print(f"{layer.name[:44]:<45} {shape_str:>15} "
                  f"{layer.entropy:>8.3f} {layer.savings_percent:>7.1f}%")
