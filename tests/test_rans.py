#!/usr/bin/env python3
"""
Unit tests for rANS codec.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantizer import quantize_affine, compute_entropy
from src.rans import (
    RANSTable, RANSEncoder, RANSDecoder,
    encode_tile, decode_tile, verify_roundtrip
)


def test_rans_table_construction():
    """Test frequency table construction."""
    counts = np.array([100, 200, 50, 150, 100, 100, 100, 100,
                       100, 100, 100, 100, 50, 50, 50, 50])
    
    table = RANSTable.from_counts(counts)
    
    # Check frequency sum (PROB_SCALE = 16384 with PROB_BITS = 14)
    assert table.freq.sum() == 16384, f"Frequencies must sum to PROB_SCALE, got {table.freq.sum()}"
    
    # Check cumulative frequencies
    assert table.cumfreq[0] == 0
    assert table.cumfreq[-1] == 16384
    
    # Check symbol table
    assert len(table.sym_table) == 16384
    assert table.sym_table.min() == 0
    assert table.sym_table.max() <= 15
    
    print("✓ RANSTable construction")


def test_rans_roundtrip_uniform():
    """Test encode/decode with uniform distribution."""
    np.random.seed(42)
    
    # Uniform distribution (worst case for compression)
    symbols = np.random.randint(0, 16, size=10000).astype(np.uint8)
    
    # Encode
    counts = np.bincount(symbols, minlength=16)
    table = RANSTable.from_counts(counts)
    
    encoder = RANSEncoder(table)
    compressed = encoder.encode(symbols)
    
    # Decode
    decoder = RANSDecoder(table)
    decoded = decoder.decode(compressed, len(symbols))
    
    # Verify
    assert np.array_equal(symbols, decoded), "Roundtrip failed!"
    
    # Check compression ratio (should be ~1x for uniform)
    original_bytes = len(symbols) // 2  # 4-bit packed
    ratio = original_bytes / len(compressed)
    assert 0.9 < ratio < 1.2, f"Unexpected ratio for uniform: {ratio}"
    
    print(f"✓ Uniform roundtrip (ratio: {ratio:.2f}x)")


def test_rans_roundtrip_gaussian():
    """Test encode/decode with Gaussian-like distribution."""
    np.random.seed(42)
    
    # Gaussian distribution (concentrated in middle)
    weights = np.random.randn(10000).astype(np.float32) * 0.02
    quantized = quantize_affine(weights, bits=4)
    symbols = quantized.indices.flatten()
    
    # Encode
    counts = np.bincount(symbols, minlength=16)
    table = RANSTable.from_counts(counts)
    
    encoder = RANSEncoder(table)
    compressed = encoder.encode(symbols)
    
    # Decode
    decoder = RANSDecoder(table)
    decoded = decoder.decode(compressed, len(symbols))
    
    # Verify
    assert np.array_equal(symbols, decoded), "Roundtrip failed!"
    
    # Check compression ratio (should be >1.3x for Gaussian)
    original_bytes = len(symbols) // 2
    ratio = original_bytes / len(compressed)
    assert ratio > 1.2, f"Expected good compression for Gaussian: {ratio}"
    
    print(f"✓ Gaussian roundtrip (ratio: {ratio:.2f}x)")


def test_encode_tile_api():
    """Test high-level encode_tile API."""
    np.random.seed(42)
    
    weights = np.random.randn(1000).astype(np.float32) * 0.02
    quantized = quantize_affine(weights, bits=4)
    
    tile = encode_tile(quantized.indices)
    
    assert tile.n_symbols == 1000
    assert tile.compression_ratio > 1.0
    assert len(tile.data) > 0
    
    print(f"✓ encode_tile API (ratio: {tile.compression_ratio:.2f}x)")


def test_decode_tile_api():
    """Test high-level decode_tile API."""
    np.random.seed(42)
    
    weights = np.random.randn(1000).astype(np.float32) * 0.02
    quantized = quantize_affine(weights, bits=4)
    
    tile = encode_tile(quantized.indices)
    decoded = decode_tile(tile)
    
    assert np.array_equal(quantized.indices.flatten(), decoded)
    
    print("✓ decode_tile API")


def test_verify_roundtrip():
    """Test verify_roundtrip utility."""
    np.random.seed(42)
    
    # Should pass
    weights = np.random.randn(1000).astype(np.float32) * 0.02
    quantized = quantize_affine(weights, bits=4)
    assert verify_roundtrip(quantized.indices), "Roundtrip should pass"
    
    print("✓ verify_roundtrip")


def test_table_serialization():
    """Test table serialization/deserialization."""
    counts = np.array([100, 200, 50, 150, 100, 100, 100, 100,
                       100, 100, 100, 100, 50, 50, 50, 50])
    
    table = RANSTable.from_counts(counts)
    
    # Serialize
    data = table.to_bytes()
    
    # Deserialize
    table2 = RANSTable.from_bytes(data)
    
    # Compare
    assert np.array_equal(table.freq, table2.freq)
    assert np.array_equal(table.cumfreq, table2.cumfreq)
    
    print("✓ Table serialization")


def test_large_tensor():
    """Test with large tensor (1M elements)."""
    np.random.seed(42)
    
    weights = np.random.randn(1_000_000).astype(np.float32) * 0.02
    quantized = quantize_affine(weights, bits=4)
    
    assert verify_roundtrip(quantized.indices), "Large tensor roundtrip failed"
    
    tile = encode_tile(quantized.indices)
    print(f"✓ Large tensor (1M elements, ratio: {tile.compression_ratio:.2f}x)")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print(" rANS Codec Tests")
    print("="*60)
    
    test_rans_table_construction()
    test_rans_roundtrip_uniform()
    test_rans_roundtrip_gaussian()
    test_encode_tile_api()
    test_decode_tile_api()
    test_verify_roundtrip()
    test_table_serialization()
    test_large_tensor()
    
    print("\n" + "="*60)
    print(" All tests passed!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
