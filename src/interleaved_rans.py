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
    """Compressed tile with interleaved streams."""
    data: bytes           # Concatenated stream data
    stream_offsets: List[int]  # Offset of each stream in data
    n_symbols: int
    n_streams: int
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
    
    def to_bytes(self) -> bytes:
        """Serialize for Metal kernel."""
        # Header: n_streams (1B), n_symbols (4B), stream_lengths (n_streams * 4B)
        stream_lengths = []
        for i in range(self.n_streams):
            if i + 1 < len(self.stream_offsets):
                stream_lengths.append(self.stream_offsets[i + 1] - self.stream_offsets[i])
            else:
                stream_lengths.append(len(self.data) - self.stream_offsets[i])
        
        header = struct.pack(
            f'<BI{self.n_streams}I',
            self.n_streams,
            self.n_symbols,
            *stream_lengths
        )
        return header + self.table.to_bytes() + self.data


def interleaved_rans_encode(symbols: np.ndarray, table: RANSTable, n_streams: int = 4) -> InterleavedEncodedTile:
    """
    Encode symbols using interleaved rANS streams.
    
    Each stream encodes every n_streams-th symbol.
    """
    symbols = symbols.flatten().astype(np.uint32)
    n = len(symbols)
    
    # Split symbols into interleaved streams
    stream_symbols = [symbols[i::n_streams] for i in range(n_streams)]
    
    # Encode each stream independently
    stream_data = []
    stream_offsets = []
    
    current_offset = 0
    for stream_idx in range(n_streams):
        stream_offsets.append(current_offset)
        
        syms = stream_symbols[stream_idx]
        if len(syms) == 0:
            stream_data.append(b'')
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
        stream_data.append(encoded)
        current_offset += len(encoded)
    
    # Concatenate all streams
    all_data = b''.join(stream_data)
    
    return InterleavedEncodedTile(
        data=all_data,
        stream_offsets=stream_offsets,
        n_symbols=n,
        n_streams=n_streams,
        table=table,
        original_bytes=(n * 4 + 7) // 8  # 4-bit packed
    )


def interleaved_rans_decode(tile: InterleavedEncodedTile) -> np.ndarray:
    """
    Decode interleaved rANS streams.
    
    This Python version is sequential for verification.
    The Metal version processes all streams in parallel.
    """
    data = tile.data
    n_streams = tile.n_streams
    n_symbols = tile.n_symbols
    table = tile.table
    
    # Calculate symbols per stream
    symbols_per_stream = (n_symbols + n_streams - 1) // n_streams
    
    # Decode each stream
    output = np.zeros(n_symbols, dtype=np.uint8)
    
    for stream_idx in range(n_streams):
        # Get stream data slice
        start = tile.stream_offsets[stream_idx]
        if stream_idx + 1 < n_streams:
            end = tile.stream_offsets[stream_idx + 1]
        else:
            end = len(data)
        
        stream_data = list(data[start:end])
        if len(stream_data) < 4:
            continue
        
        # Initialize state (4 bytes, big-endian)
        ptr = 0
        state = (stream_data[ptr] << 24) | (stream_data[ptr+1] << 16) | \
                (stream_data[ptr+2] << 8) | stream_data[ptr+3]
        ptr += 4
        
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
            
            # Renormalize
            while state < RANS_BYTE_L and ptr < len(stream_data):
                state = (state << 8) | stream_data[ptr]
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
