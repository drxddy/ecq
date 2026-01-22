"""
rANS (Range Asymmetric Numeral Systems) Entropy Codec

A correct implementation based on Fabian Giesen's reference.

References:
- https://github.com/rygorous/ryg_rans
- Duda, J. (2014). Asymmetric numeral systems.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import struct


# ============================================================================
# Constants - matching ryg_rans reference
# ============================================================================

PROB_BITS = 14          # Probability precision
PROB_SCALE = 1 << PROB_BITS  # 16384
RANS_BYTE_L = 1 << 23   # Lower bound for state


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RANSTable:
    """Precomputed tables for rANS."""
    freq: np.ndarray
    cumfreq: np.ndarray  
    sym_table: np.ndarray
    n_symbols: int
    
    @classmethod
    def from_counts(cls, counts: np.ndarray) -> 'RANSTable':
        """Build tables from histogram."""
        n_symbols = len(counts)
        counts = np.maximum(counts, 1).astype(np.float64)
        
        # Scale to PROB_SCALE
        freq = (counts / counts.sum() * PROB_SCALE).astype(np.uint32)
        
        # Adjust for rounding
        diff = PROB_SCALE - freq.sum()
        freq[np.argmax(freq)] += diff
        
        # Cumulative
        cumfreq = np.zeros(n_symbols + 1, dtype=np.uint32)
        cumfreq[1:] = np.cumsum(freq)
        
        # Symbol lookup
        sym_table = np.zeros(PROB_SCALE, dtype=np.uint8)
        for s in range(n_symbols):
            sym_table[cumfreq[s]:cumfreq[s+1]] = s
        
        return cls(freq=freq, cumfreq=cumfreq, sym_table=sym_table, n_symbols=n_symbols)
    
    def to_bytes(self) -> bytes:
        """Serialize."""
        return struct.pack(f'<H{self.n_symbols}H', self.n_symbols, *self.freq.tolist())
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'RANSTable':
        """Deserialize."""
        n_symbols = struct.unpack('<H', data[:2])[0]
        freq = np.array(struct.unpack(f'<{n_symbols}H', data[2:2+2*n_symbols]), dtype=np.uint32)
        return cls.from_counts(freq)


@dataclass
class EncodedTile:
    """Compressed tile with metadata."""
    data: bytes
    n_symbols: int
    table: RANSTable
    original_bytes: int
    
    @property
    def compressed_bytes(self) -> int:
        return len(self.data)
    
    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / self.compressed_bytes if self.compressed_bytes > 0 else float('inf')
    
    @property
    def bits_per_symbol(self) -> float:
        return (self.compressed_bytes * 8) / self.n_symbols if self.n_symbols > 0 else 0


# ============================================================================
# Encoder - following ryg_rans exactly
# ============================================================================

def rans_encode(symbols: np.ndarray, table: RANSTable) -> bytes:
    """
    Encode symbols using rANS.
    
    rANS encoding formula:
        C(x, s) = (x // freq[s]) * M + cumfreq[s] + (x % freq[s])
    where M = PROB_SCALE
    """
    symbols = symbols.flatten().astype(np.uint32)
    n = len(symbols)
    
    # Output bytes (we'll reverse at the end)
    out_bytes = []
    
    # State
    state = RANS_BYTE_L
    
    # Encode in reverse
    for i in range(n - 1, -1, -1):
        s = symbols[i]
        freq_s = int(table.freq[s])
        start_s = int(table.cumfreq[s])
        
        # Renormalize
        x_max = ((RANS_BYTE_L >> PROB_BITS) << 8) * freq_s
        while state >= x_max:
            out_bytes.append(state & 0xFF)
            state >>= 8
        
        # Encode
        state = ((state // freq_s) << PROB_BITS) + (state % freq_s) + start_s
    
    # Flush state (4 bytes, big-endian)
    out_bytes.append((state >> 0) & 0xFF)
    out_bytes.append((state >> 8) & 0xFF)
    out_bytes.append((state >> 16) & 0xFF)
    out_bytes.append((state >> 24) & 0xFF)
    
    # Reverse for decoding
    return bytes(reversed(out_bytes))


# ============================================================================
# Decoder - following ryg_rans exactly  
# ============================================================================

def rans_decode(data: bytes, n_symbols: int, table: RANSTable) -> np.ndarray:
    """
    Decode symbols using rANS.
    
    rANS decoding:
        slot = x & (M - 1)
        s = symbol_table[slot]
        x = freq[s] * (x >> PROB_BITS) + slot - cumfreq[s]
    """
    data = list(data)
    ptr = 0
    
    # Initialize state (4 bytes, big-endian)
    state = (data[ptr] << 24) | (data[ptr+1] << 16) | (data[ptr+2] << 8) | data[ptr+3]
    ptr += 4
    
    output = np.zeros(n_symbols, dtype=np.uint8)
    
    for i in range(n_symbols):
        # Get slot
        slot = state & (PROB_SCALE - 1)
        
        # Lookup symbol
        s = int(table.sym_table[slot])
        output[i] = s
        
        # Decode step
        freq_s = int(table.freq[s])
        start_s = int(table.cumfreq[s])
        state = freq_s * (state >> PROB_BITS) + slot - start_s
        
        # Renormalize
        while state < RANS_BYTE_L and ptr < len(data):
            state = (state << 8) | data[ptr]
            ptr += 1
    
    return output


# ============================================================================
# High-Level API
# ============================================================================

class RANSEncoder:
    def __init__(self, table: RANSTable):
        self.table = table
    
    def encode(self, symbols: np.ndarray) -> bytes:
        return rans_encode(symbols, self.table)


class RANSDecoder:
    def __init__(self, table: RANSTable):
        self.table = table
    
    def decode(self, data: bytes, n_symbols: int) -> np.ndarray:
        return rans_decode(data, n_symbols, self.table)


def encode_tile(indices: np.ndarray, bits: int = 4) -> EncodedTile:
    """Encode indices to compressed tile."""
    indices = indices.flatten()
    n_symbols = 1 << bits
    
    counts = np.bincount(indices, minlength=n_symbols)
    table = RANSTable.from_counts(counts)
    
    compressed = rans_encode(indices, table)
    original_bytes = (len(indices) * bits + 7) // 8
    
    return EncodedTile(
        data=compressed,
        n_symbols=len(indices),
        table=table,
        original_bytes=original_bytes
    )


def decode_tile(tile: EncodedTile) -> np.ndarray:
    """Decode compressed tile to indices."""
    return rans_decode(tile.data, tile.n_symbols, tile.table)


def verify_roundtrip(indices: np.ndarray, bits: int = 4) -> bool:
    """Verify lossless roundtrip."""
    tile = encode_tile(indices, bits)
    decoded = decode_tile(tile)
    return np.array_equal(indices.flatten(), decoded)
