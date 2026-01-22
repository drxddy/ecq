"""
Interleaved rANS Codec for Parallel Decoding

This implementation uses multiple independent rANS streams that can be
decoded in parallel, enabling SIMD vectorization on GPU.

Interleaving pattern (4 streams):
  Stream 0: symbols 0, 4, 8, 12, ...
  Stream 1: symbols 1, 5, 9, 13, ...
  Stream 2: symbols 2, 6, 10, 14, ...
  Stream 3: symbols 3, 7, 11, 15, ...

References:
- Fabian Giesen, "Interleaved entropy coders" (2014)
- https://fgiesen.wordpress.com/2014/02/18/rans-with-static-probability-distributions/
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import struct

from .rans import PROB_BITS, PROB_SCALE, RANS_BYTE_L, RANSTable


@dataclass
class InterleavedEncodedTile:
    """Compressed tile with physically interleaved streams.
    
    Data layout (CRITICAL for GPU coalescing):
    - Old (logical): [Stream0 bytes...][Stream1 bytes...][Stream2 bytes...]
    - New (physical): [S0_B0, S1_B0, S2_B0, S3_B0, S0_B1, S1_B1, S2_B1, S3_B1, ...]
    
    This ensures adjacent GPU threads read adjacent memory addresses.
    """
    data: bytes           # Physically interleaved stream data
    stream_lengths: List[int]  # Length of each stream (for decode bounds)
    max_stream_len: int   # Padded length (all streams same size for interleaving)
    n_symbols: int
    n_streams: int
    table: RANSTable
    original_bytes: int
    
    @property
    def stream_offsets(self) -> List[int]:
        """Compatibility: compute logical offsets from physical layout."""
        # In physical layout, each stream is at stride n_streams
        # stream_offsets[i] = i (byte offset within each interleaved chunk)
        return list(range(self.n_streams))
    
    @property
    def compressed_bytes(self) -> int:
        return len(self.data)
    
    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / self.compressed_bytes if self.compressed_bytes > 0 else float('inf')
    
    @property
    def bits_per_symbol(self) -> float:
        return (self.compressed_bytes * 8) / self.n_symbols if self.n_symbols > 0 else 0
    
    def to_bytes(self) -> bytes:
        """Serialize for Metal kernel."""
        # Header: n_streams (1B), n_symbols (4B), max_stream_len (4B), stream_lengths (n_streams * 4B)
        header = struct.pack(
            f'<BII{self.n_streams}I',
            self.n_streams,
            self.n_symbols,
            self.max_stream_len,
            *self.stream_lengths
        )
        return header + self.table.to_bytes() + self.data


def interleaved_rans_encode(symbols: np.ndarray, table: RANSTable, n_streams: int = 4) -> InterleavedEncodedTile:
    """
    Encode symbols using interleaved rANS streams with PHYSICAL interleaving.
    
    Each stream encodes every n_streams-th symbol.
    Output data is physically interleaved for GPU memory coalescing:
    [S0_B0, S1_B0, S2_B0, S3_B0, S0_B1, S1_B1, ...]
    """
    symbols = symbols.flatten().astype(np.uint32)
    n = len(symbols)
    
    # Split symbols into interleaved streams
    stream_symbols = [symbols[i::n_streams] for i in range(n_streams)]
    
    # Encode each stream independently (get raw bytes per stream)
    stream_bytes_list = []
    stream_lengths = []
    
    for stream_idx in range(n_streams):
        syms = stream_symbols[stream_idx]
        if len(syms) == 0:
            stream_bytes_list.append(b'')
            stream_lengths.append(0)
            continue
        
        # Encode this stream
        out_bytes = []
        state = RANS_BYTE_L
        
        # Encode in reverse order
        for i in range(len(syms) - 1, -1, -1):
            s = syms[i]
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
        
        # Reverse for decoding order
        encoded = bytes(reversed(out_bytes))
        stream_bytes_list.append(encoded)
        stream_lengths.append(len(encoded))
    
    # PHYSICAL INTERLEAVING (Critical for GPU coalescing)
    # Pad all streams to same length, then interleave byte-by-byte
    max_stream_len = max(stream_lengths) if stream_lengths else 0
    
    # Create padded numpy array for efficient interleaving
    stream_matrix = np.zeros((n_streams, max_stream_len), dtype=np.uint8)
    for i, stream_data in enumerate(stream_bytes_list):
        if len(stream_data) > 0:
            stream_matrix[i, :len(stream_data)] = np.frombuffer(stream_data, dtype=np.uint8)
    
    # Transpose and flatten: (n_streams, max_len) -> (max_len, n_streams) -> flat
    # Result: [S0_B0, S1_B0, S2_B0, S3_B0, S0_B1, S1_B1, S2_B1, S3_B1, ...]
    interleaved_data = stream_matrix.T.flatten().tobytes()
    
    return InterleavedEncodedTile(
        data=interleaved_data,
        stream_lengths=stream_lengths,
        max_stream_len=max_stream_len,
        n_symbols=n,
        n_streams=n_streams,
        table=table,
        original_bytes=(n * 4 + 7) // 8  # 4-bit packed
    )


def interleaved_rans_decode(tile: InterleavedEncodedTile) -> np.ndarray:
    """
    Decode physically interleaved rANS streams.
    
    Data layout: [S0_B0, S1_B0, S2_B0, S3_B0, S0_B1, S1_B1, ...]
    Access pattern: data[stream_idx + ptr * n_streams]
    
    This Python version is sequential for verification.
    The Metal version processes all streams in parallel with coalesced reads.
    """
    data = np.frombuffer(tile.data, dtype=np.uint8)
    n_streams = tile.n_streams
    n_symbols = tile.n_symbols
    max_stream_len = tile.max_stream_len
    stream_lengths = tile.stream_lengths
    table = tile.table
    
    # Decode each stream
    output = np.zeros(n_symbols, dtype=np.uint8)
    
    for stream_idx in range(n_streams):
        stream_len = stream_lengths[stream_idx]
        if stream_len < 4:
            continue
        
        # COALESCED READ PATTERN: data[stream_idx + ptr * n_streams]
        def read_byte(ptr: int) -> int:
            return int(data[stream_idx + ptr * n_streams])
        
        # Initialize state (4 bytes, big-endian) - coalesced reads
        ptr = 0
        state = (read_byte(0) << 24) | (read_byte(1) << 16) | \
                (read_byte(2) << 8) | read_byte(3)
        ptr = 4
        
        # Number of symbols in this stream
        n_syms = len(range(stream_idx, n_symbols, n_streams))
        
        # Decode symbols for this stream
        for i in range(n_syms):
            output_idx = stream_idx + i * n_streams
            if output_idx >= n_symbols:
                break
            
            # Get slot
            slot = state & (PROB_SCALE - 1)
            
            # Lookup symbol
            s = int(table.sym_table[slot])
            output[output_idx] = s
            
            # Decode step
            freq_s = int(table.freq[s])
            start_s = int(table.cumfreq[s])
            state = freq_s * (state >> PROB_BITS) + slot - start_s
            
            # Renormalize (coalesced reads)
            while state < RANS_BYTE_L and ptr < stream_len:
                state = (state << 8) | read_byte(ptr)
                ptr += 1
    
    return output


def verify_interleaved_roundtrip(indices: np.ndarray, n_streams: int = 4, bits: int = 4) -> Tuple[bool, float]:
    """Verify lossless roundtrip and return compression ratio."""
    indices = indices.flatten()
    n_symbols = 1 << bits
    
    # Build frequency table
    counts = np.bincount(indices, minlength=n_symbols)
    table = RANSTable.from_counts(counts)
    
    # Encode
    tile = interleaved_rans_encode(indices, table, n_streams)
    
    # Decode
    decoded = interleaved_rans_decode(tile)
    
    # Verify
    matches = np.array_equal(indices, decoded)
    
    return matches, tile.compression_ratio


class InterleavedRANSEncoder:
    """High-level encoder API."""
    
    def __init__(self, n_streams: int = 4, bits: int = 4):
        self.n_streams = n_streams
        self.bits = bits
    
    def encode(self, indices: np.ndarray) -> InterleavedEncodedTile:
        """Encode quantization indices."""
        indices = indices.flatten()
        n_symbols = 1 << self.bits
        
        counts = np.bincount(indices, minlength=n_symbols)
        table = RANSTable.from_counts(counts)
        
        return interleaved_rans_encode(indices, table, self.n_streams)


class InterleavedRANSDecoder:
    """High-level decoder API."""
    
    def decode(self, tile: InterleavedEncodedTile) -> np.ndarray:
        return interleaved_rans_decode(tile)


if __name__ == "__main__":
    # Test with random 4-bit indices
    np.random.seed(42)
    
    # Generate Gaussian-like distribution (what we see in LLM weights)
    raw = np.random.randn(10000)
    indices = ((raw - raw.min()) / (raw.max() - raw.min()) * 15).astype(np.uint8)
    indices = np.clip(indices, 0, 15)
    
    print("Testing Interleaved rANS Codec")
    print("="*50)
    
    for n_streams in [1, 2, 4, 8, 16]:
        ok, ratio = verify_interleaved_roundtrip(indices, n_streams)
        status = "✓" if ok else "✗"
        print(f"  {n_streams} streams: {status} compression={ratio:.2f}x, "
              f"bits/sym={4/ratio:.2f}")
