#!/usr/bin/env python3
"""
Entropy-Coded 4-bit Quantization Prototype

Key insight: Near-Gaussian weights → non-uniform 4-bit symbol distribution
→ Shannon entropy < 4 bits → rANS can compress further

This exploits DISTRIBUTIONAL redundancy, not STRUCTURAL redundancy.
"""

import numpy as np
from collections import Counter
import time
from typing import Tuple, Dict
import struct

# =============================================================================
# PART 1: Entropy Analysis - How much can we save?
# =============================================================================

def quantize_4bit(weights: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Standard 4-bit affine quantization."""
    w_min, w_max = weights.min(), weights.max()
    scale = (w_max - w_min) / 15
    if scale == 0:
        scale = 1e-8
    indices = np.clip(np.round((weights - w_min) / scale), 0, 15).astype(np.uint8)
    return indices, scale, w_min

def dequantize_4bit(indices: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
    """Reconstruct from 4-bit indices."""
    return indices.astype(np.float32) * scale + zero_point

def compute_entropy(indices: np.ndarray) -> float:
    """Compute Shannon entropy in bits per symbol."""
    counts = Counter(indices.flatten())
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

def analyze_entropy_savings(weights: np.ndarray, name: str = "weights") -> Dict:
    """Analyze potential entropy coding savings."""
    indices, scale, zero_point = quantize_4bit(weights)
    entropy = compute_entropy(indices)
    
    # Calculate savings
    fixed_bits = 4.0
    savings_percent = (1 - entropy / fixed_bits) * 100
    
    # Symbol distribution
    counts = Counter(indices.flatten())
    total = sum(counts.values())
    dist = {k: counts.get(k, 0) / total for k in range(16)}
    
    result = {
        "name": name,
        "shape": weights.shape,
        "entropy_bits": entropy,
        "fixed_bits": fixed_bits,
        "savings_percent": savings_percent,
        "symbol_distribution": dist,
        "theoretical_compression": fixed_bits / entropy if entropy > 0 else float('inf'),
    }
    
    print(f"\n{'='*60}")
    print(f"Entropy Analysis: {name}")
    print(f"{'='*60}")
    print(f"Shape: {weights.shape} ({weights.size:,} elements)")
    print(f"Shannon Entropy: {entropy:.3f} bits/symbol")
    print(f"Fixed 4-bit:     4.000 bits/symbol")
    print(f"Potential Savings: {savings_percent:.1f}%")
    print(f"Effective Compression: {result['theoretical_compression']:.2f}x over 4-bit")
    print(f"\nSymbol Distribution (should be bell-curved for Gaussian):")
    
    # ASCII histogram
    max_prob = max(dist.values())
    for sym in range(16):
        bar_len = int(40 * dist[sym] / max_prob) if max_prob > 0 else 0
        print(f"  {sym:2d}: {'█' * bar_len} {dist[sym]*100:.1f}%")
    
    return result


# =============================================================================
# PART 2: rANS (Range Asymmetric Numeral Systems) Implementation
# =============================================================================

class RANSEncoder:
    """
    Simple rANS encoder for 4-bit symbols.
    
    rANS state equation:
        x' = (x // freq[s]) * total_freq + cumfreq[s] + (x % freq[s])
    """
    
    def __init__(self, freq_table: np.ndarray, precision_bits: int = 12):
        self.precision_bits = precision_bits
        self.total_freq = 1 << precision_bits  # 4096
        
        # Normalize frequencies to sum to total_freq
        freq_table = np.maximum(freq_table, 1)  # Avoid zero frequencies
        self.freq = (freq_table / freq_table.sum() * self.total_freq).astype(np.int32)
        
        # Ensure sum equals total_freq
        diff = self.total_freq - self.freq.sum()
        self.freq[np.argmax(self.freq)] += diff
        
        # Cumulative frequencies
        self.cumfreq = np.zeros(17, dtype=np.int32)
        self.cumfreq[1:] = np.cumsum(self.freq)
        
    def encode(self, symbols: np.ndarray) -> bytes:
        """Encode symbols to compressed bytes."""
        # rANS encodes in reverse order
        symbols = symbols.flatten()[::-1]
        
        state = 1 << 31  # Initial state
        output = []
        
        L = 1 << 23  # Lower bound for renormalization
        
        for s in symbols:
            freq_s = self.freq[s]
            cumfreq_s = self.cumfreq[s]
            
            # Renormalize: output bytes while state is too large
            while state >= L * freq_s:
                output.append(state & 0xFF)
                state >>= 8
            
            # Core rANS step
            state = (state // freq_s) * self.total_freq + cumfreq_s + (state % freq_s)
        
        # Flush remaining state (4 bytes)
        for _ in range(4):
            output.append(state & 0xFF)
            state >>= 8
            
        return bytes(output[::-1])  # Reverse for decoding order
    
    
class RANSDecoder:
    """Simple rANS decoder."""
    
    def __init__(self, freq_table: np.ndarray, precision_bits: int = 12):
        self.precision_bits = precision_bits
        self.total_freq = 1 << precision_bits
        
        # Same frequency normalization as encoder
        freq_table = np.maximum(freq_table, 1)
        self.freq = (freq_table / freq_table.sum() * self.total_freq).astype(np.int32)
        diff = self.total_freq - self.freq.sum()
        self.freq[np.argmax(self.freq)] += diff
        
        self.cumfreq = np.zeros(17, dtype=np.int32)
        self.cumfreq[1:] = np.cumsum(self.freq)
        
        # Build lookup table for fast symbol recovery
        self.symbol_table = np.zeros(self.total_freq, dtype=np.uint8)
        for s in range(16):
            self.symbol_table[self.cumfreq[s]:self.cumfreq[s+1]] = s
    
    def decode(self, data: bytes, num_symbols: int) -> np.ndarray:
        """Decode compressed bytes back to symbols."""
        data = list(data)
        ptr = 0
        
        # Initialize state from first 4 bytes
        state = 0
        for _ in range(4):
            state = (state << 8) | data[ptr]
            ptr += 1
        
        output = []
        mask = self.total_freq - 1
        
        for _ in range(num_symbols):
            # Extract slot from state
            slot = state & mask
            
            # Lookup symbol
            s = self.symbol_table[slot]
            output.append(s)
            
            # Update state
            freq_s = self.freq[s]
            cumfreq_s = self.cumfreq[s]
            state = freq_s * (state >> self.precision_bits) + slot - cumfreq_s
            
            # Renormalize: read bytes while state is too small
            while state < (1 << 23) and ptr < len(data):
                state = (state << 8) | data[ptr]
                ptr += 1
        
        return np.array(output, dtype=np.uint8)


# =============================================================================
# PART 3: End-to-End Compression Test
# =============================================================================

def test_rans_compression(weights: np.ndarray, name: str = "test") -> Dict:
    """Full encode/decode cycle with timing."""
    
    print(f"\n{'='*60}")
    print(f"rANS Compression Test: {name}")
    print(f"{'='*60}")
    
    # Quantize
    indices, scale, zero_point = quantize_4bit(weights)
    original_size = indices.size  # In 4-bit units
    original_bytes = (indices.size + 1) // 2  # Packed 4-bit
    
    # Build frequency table from data
    counts = np.bincount(indices.flatten(), minlength=16)
    
    # Encode
    encoder = RANSEncoder(counts)
    t0 = time.perf_counter()
    compressed = encoder.encode(indices)
    encode_time = time.perf_counter() - t0
    
    compressed_bytes = len(compressed)
    
    # Decode
    decoder = RANSDecoder(counts)
    t0 = time.perf_counter()
    decoded = decoder.decode(compressed, indices.size)
    decode_time = time.perf_counter() - t0
    
    # Verify lossless
    decoded = decoded.reshape(indices.shape)
    is_lossless = np.array_equal(indices, decoded)
    
    # Dequantize and check
    reconstructed = dequantize_4bit(decoded, scale, zero_point)
    mse = np.mean((weights - reconstructed) ** 2)
    sqnr = 10 * np.log10(np.var(weights) / mse) if mse > 0 else float('inf')
    
    # Stats
    compression_ratio = original_bytes / compressed_bytes
    bits_per_symbol = (compressed_bytes * 8) / indices.size
    entropy = compute_entropy(indices)
    overhead = bits_per_symbol - entropy  # How close to entropy?
    
    print(f"Original (4-bit packed): {original_bytes:,} bytes")
    print(f"Compressed (rANS):       {compressed_bytes:,} bytes")
    print(f"Compression ratio:       {compression_ratio:.3f}x over 4-bit")
    print(f"Bits per symbol:         {bits_per_symbol:.3f} (entropy: {entropy:.3f})")
    print(f"Coding overhead:         {overhead:.3f} bits ({overhead/entropy*100:.1f}%)")
    print(f"Lossless:                {is_lossless}")
    print(f"SQNR (vs original):      {sqnr:.2f} dB")
    print(f"Encode time:             {encode_time*1000:.2f} ms")
    print(f"Decode time:             {decode_time*1000:.2f} ms")
    print(f"Decode throughput:       {indices.size / decode_time / 1e6:.1f} M symbols/s")
    
    return {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": compression_ratio,
        "bits_per_symbol": bits_per_symbol,
        "entropy": entropy,
        "is_lossless": is_lossless,
        "sqnr_db": sqnr,
        "decode_time_ms": decode_time * 1000,
        "decode_throughput_Msym_s": indices.size / decode_time / 1e6,
    }


# =============================================================================
# PART 4: Test on Synthetic and Real Weights
# =============================================================================

def test_synthetic():
    """Test on synthetic Gaussian weights."""
    print("\n" + "="*70)
    print("SYNTHETIC GAUSSIAN WEIGHTS")
    print("="*70)
    
    np.random.seed(42)
    
    # Simulate a 4096x4096 weight matrix (like LLaMA MLP)
    weights = np.random.randn(4096, 4096).astype(np.float32) * 0.02
    
    analyze_entropy_savings(weights, "Gaussian 4096x4096")
    test_rans_compression(weights, "Gaussian 4096x4096")


def test_real_model():
    """Test on real model weights if available."""
    print("\n" + "="*70)
    print("REAL MODEL WEIGHTS")  
    print("="*70)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM
        
        print("Loading SmolLM-135M...")
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM-135M",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        results = []
        
        # Test different layer types
        for name, param in model.named_parameters():
            if param.ndim == 2 and param.numel() > 100000:  # Only big matrices
                weights = param.detach().numpy()
                result = analyze_entropy_savings(weights, name)
                results.append(result)
                
                if len(results) >= 5:  # Test first 5 big matrices
                    break
        
        # Full compression test on one matrix
        for name, param in model.named_parameters():
            if "down_proj" in name and param.ndim == 2:
                weights = param.detach().numpy()
                test_rans_compression(weights, name)
                break
                
        # Summary
        if results:
            avg_entropy = np.mean([r["entropy_bits"] for r in results])
            avg_savings = np.mean([r["savings_percent"] for r in results])
            print(f"\n{'='*60}")
            print("SUMMARY ACROSS LAYERS")
            print(f"{'='*60}")
            print(f"Average entropy:  {avg_entropy:.3f} bits")
            print(f"Average savings:  {avg_savings:.1f}% over fixed 4-bit")
            print(f"At 150 GB/s bandwidth, this means:")
            print(f"  - Fixed 4-bit: 150 GB/s")
            print(f"  - Entropy-coded: {150 * (1 + avg_savings/100):.1f} GB/s effective")
                
    except ImportError:
        print("torch/transformers not available, skipping real model test")
        print("Install with: pip install torch transformers")


def analyze_bandwidth_impact():
    """Calculate real-world bandwidth savings on Apple Silicon."""
    print("\n" + "="*70)
    print("BANDWIDTH IMPACT ANALYSIS (Apple Silicon M3 Pro)")
    print("="*70)
    
    # Typical values from experiments
    avg_entropy = 3.4  # bits per symbol (typical for Gaussian weights)
    fixed_bits = 4.0
    
    bandwidth = 150  # GB/s
    model_size_4bit = 3.5  # GB for 7B model at 4-bit
    
    savings_ratio = fixed_bits / avg_entropy
    effective_bandwidth = bandwidth * savings_ratio
    
    tokens_4bit = bandwidth / model_size_4bit
    tokens_entropy = effective_bandwidth / model_size_4bit
    
    print(f"Model size (4-bit):     {model_size_4bit} GB")
    print(f"Memory bandwidth:       {bandwidth} GB/s")
    print(f"Average entropy:        {avg_entropy:.2f} bits")
    print(f"Effective compression:  {savings_ratio:.2f}x over 4-bit")
    print(f"")
    print(f"Token generation (4-bit fixed):    {tokens_4bit:.1f} tok/s")
    print(f"Token generation (entropy-coded):  {tokens_entropy:.1f} tok/s")
    print(f"Speedup:                           {tokens_entropy/tokens_4bit:.2f}x")
    print(f"")
    print(f"⚠️  Requires fused decode+GEMM kernel to realize gains")
    print(f"    Decode must be faster than bandwidth savings!")


if __name__ == "__main__":
    test_synthetic()
    test_real_model()
    analyze_bandwidth_impact()
