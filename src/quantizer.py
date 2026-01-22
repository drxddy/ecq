"""
Quantization utilities for weight compression.

Implements standard affine quantization with entropy analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    bits: int = 4
    symmetric: bool = False
    per_channel: bool = False


@dataclass  
class QuantizedTensor:
    """Quantized tensor with metadata."""
    indices: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray
    shape: Tuple[int, ...]
    bits: int
    
    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))
    
    def dequantize(self) -> np.ndarray:
        """Reconstruct float tensor from quantized representation."""
        return self.indices.astype(np.float32) * self.scale + self.zero_point


def quantize_affine(
    weights: np.ndarray,
    bits: int = 4,
    per_channel: bool = False,
    channel_axis: int = 0
) -> QuantizedTensor:
    """
    Affine (asymmetric) quantization.
    
    Args:
        weights: Float tensor to quantize
        bits: Number of bits (default 4)
        per_channel: If True, compute scale/zero per channel
        channel_axis: Axis for per-channel quantization
        
    Returns:
        QuantizedTensor with indices and metadata
    """
    max_val = (1 << bits) - 1  # 15 for 4-bit
    
    if per_channel:
        # Compute per-channel statistics
        axes = tuple(i for i in range(weights.ndim) if i != channel_axis)
        w_min = weights.min(axis=axes, keepdims=True)
        w_max = weights.max(axis=axes, keepdims=True)
    else:
        w_min = weights.min()
        w_max = weights.max()
    
    # Compute scale and zero point
    scale = (w_max - w_min) / max_val
    scale = np.where(scale == 0, 1e-8, scale)  # Avoid division by zero
    zero_point = w_min
    
    # Quantize
    indices = np.clip(
        np.round((weights - zero_point) / scale),
        0, max_val
    ).astype(np.uint8)
    
    return QuantizedTensor(
        indices=indices,
        scale=scale.astype(np.float32),
        zero_point=zero_point.astype(np.float32),
        shape=weights.shape,
        bits=bits
    )


def compute_sqnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Compute Signal-to-Quantization-Noise Ratio in dB.
    
    SQNR = 10 * log10(signal_power / noise_power)
    """
    signal_power = np.var(original)
    noise_power = np.mean((original - reconstructed) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)


def compute_entropy(indices: np.ndarray, bits: int = 4) -> float:
    """
    Compute Shannon entropy of quantized indices in bits per symbol.
    
    For uniform distribution, entropy = bits.
    For concentrated distribution, entropy < bits.
    """
    n_symbols = 1 << bits
    counts = np.bincount(indices.flatten(), minlength=n_symbols)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    
    return -np.sum(probs * np.log2(probs))


def compute_compression_potential(indices: np.ndarray, bits: int = 4) -> dict:
    """
    Analyze compression potential of quantized indices.
    
    Returns:
        Dictionary with entropy, savings, and theoretical compression ratio.
    """
    entropy = compute_entropy(indices, bits)
    savings_percent = (1 - entropy / bits) * 100
    compression_ratio = bits / entropy if entropy > 0 else float('inf')
    
    # Symbol distribution
    n_symbols = 1 << bits
    counts = np.bincount(indices.flatten(), minlength=n_symbols)
    probs = counts / counts.sum()
    
    return {
        'entropy': entropy,
        'fixed_bits': bits,
        'savings_percent': savings_percent,
        'compression_ratio': compression_ratio,
        'symbol_distribution': probs.tolist(),
    }
