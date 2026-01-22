"""
Compressed model format for entropy-coded quantized weights.

Defines the on-disk format and serialization for entropy-coded models.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, BinaryIO
import struct
import json

from .quantizer import quantize_affine, QuantizedTensor
from .rans import encode_tile, decode_tile, RANSTable, EncodedTile


# ============================================================================
# File Format Specification
# ============================================================================

"""
Entropy-Coded Model Format (.ecq)

Header (64 bytes):
  - Magic: 4 bytes "ECQ1"
  - Version: 4 bytes (uint32)
  - Num tensors: 4 bytes (uint32)
  - Metadata offset: 8 bytes (uint64)
  - Data offset: 8 bytes (uint64)
  - Reserved: 36 bytes

Metadata Section (JSON):
  - Tensor names, shapes, dtypes
  - Quantization parameters (scale, zero_point)
  - Tile information

Data Section:
  - For each tensor:
    - Frequency table
    - Compressed bitstream
"""

MAGIC = b'ECQ1'
VERSION = 1


@dataclass
class TensorMetadata:
    """Metadata for a single compressed tensor."""
    name: str
    shape: List[int]
    dtype: str
    bits: int
    
    # Quantization params
    scale: float
    zero_point: float
    
    # Compression info
    data_offset: int
    data_size: int
    table_offset: int
    table_size: int
    n_elements: int
    
    # Stats
    entropy: float
    compression_ratio: float


@dataclass  
class CompressedModel:
    """In-memory representation of a compressed model."""
    tensors: Dict[str, TensorMetadata]
    data: bytes
    
    @property
    def total_params(self) -> int:
        return sum(t.n_elements for t in self.tensors.values())
    
    @property
    def compressed_size(self) -> int:
        return len(self.data)
    
    @property
    def original_4bit_size(self) -> int:
        return (self.total_params + 1) // 2
    
    @property
    def compression_ratio(self) -> float:
        return self.original_4bit_size / self.compressed_size


class ModelCompressor:
    """
    Compress a model using entropy-coded quantization.
    """
    
    def __init__(self, bits: int = 4, tile_size: int = 0):
        """
        Args:
            bits: Quantization bit width
            tile_size: Tile size for parallel encoding (0 = whole tensor)
        """
        self.bits = bits
        self.tile_size = tile_size
    
    def compress_tensor(
        self,
        name: str,
        weights: np.ndarray,
    ) -> tuple:
        """
        Compress a single tensor.
        
        Returns:
            (TensorMetadata, table_bytes, data_bytes)
        """
        # Quantize
        quantized = quantize_affine(weights, bits=self.bits)
        
        # Encode
        tile = encode_tile(quantized.indices, bits=self.bits)
        
        # Serialize table
        table_bytes = tile.table.to_bytes()
        data_bytes = tile.data
        
        metadata = TensorMetadata(
            name=name,
            shape=list(weights.shape),
            dtype=str(weights.dtype),
            bits=self.bits,
            scale=float(quantized.scale.flat[0]),
            zero_point=float(quantized.zero_point.flat[0]),
            data_offset=0,  # Set later
            data_size=len(data_bytes),
            table_offset=0,  # Set later
            table_size=len(table_bytes),
            n_elements=weights.size,
            entropy=tile.bits_per_symbol,
            compression_ratio=tile.compression_ratio,
        )
        
        return metadata, table_bytes, data_bytes
    
    def compress_safetensors(
        self,
        input_path: Path,
        output_path: Path,
        progress: bool = True,
    ) -> CompressedModel:
        """
        Compress a safetensors model file.
        
        Args:
            input_path: Path to input model.safetensors
            output_path: Path to output .ecq file
            progress: Show progress bar
            
        Returns:
            CompressedModel with compression statistics
        """
        from safetensors import safe_open
        
        tensors = {}
        all_table_bytes = []
        all_data_bytes = []
        
        current_offset = 0
        
        with safe_open(input_path, framework='pt') as f:
            tensor_names = list(f.keys())
            
            for name in tensor_names:
                tensor = f.get_tensor(name)
                
                # Convert to numpy float32
                import torch
                if isinstance(tensor, torch.Tensor):
                    weights = tensor.float().numpy()
                else:
                    weights = np.array(tensor, dtype=np.float32)
                
                # Skip small tensors
                if weights.size < 1000:
                    continue
                
                # Compress
                metadata, table_bytes, data_bytes = self.compress_tensor(
                    name, weights
                )
                
                # Update offsets
                metadata.table_offset = current_offset
                current_offset += len(table_bytes)
                metadata.data_offset = current_offset
                current_offset += len(data_bytes)
                
                tensors[name] = metadata
                all_table_bytes.append(table_bytes)
                all_data_bytes.append(data_bytes)
                
                if progress:
                    ratio = metadata.compression_ratio
                    print(f"  {name[:50]:<50} {ratio:.2f}x")
        
        # Combine all data
        combined_data = b''.join(all_table_bytes) + b''.join(all_data_bytes)
        
        # Create model
        model = CompressedModel(tensors=tensors, data=combined_data)
        
        # Write to file
        self._write_file(output_path, model)
        
        return model
    
    def _write_file(self, path: Path, model: CompressedModel):
        """Write compressed model to file."""
        # Serialize metadata
        metadata_json = json.dumps({
            name: {
                'shape': t.shape,
                'dtype': t.dtype,
                'bits': t.bits,
                'scale': t.scale,
                'zero_point': t.zero_point,
                'data_offset': t.data_offset,
                'data_size': t.data_size,
                'table_offset': t.table_offset,
                'table_size': t.table_size,
                'n_elements': t.n_elements,
                'entropy': t.entropy,
                'compression_ratio': t.compression_ratio,
            }
            for name, t in model.tensors.items()
        }).encode('utf-8')
        
        # Calculate offsets
        header_size = 64
        metadata_offset = header_size
        data_offset = metadata_offset + len(metadata_json)
        
        with open(path, 'wb') as f:
            # Write header
            f.write(MAGIC)
            f.write(struct.pack('<I', VERSION))
            f.write(struct.pack('<I', len(model.tensors)))
            f.write(struct.pack('<Q', metadata_offset))
            f.write(struct.pack('<Q', data_offset))
            f.write(b'\x00' * 36)  # Reserved
            
            # Write metadata
            f.write(metadata_json)
            
            # Write data
            f.write(model.data)


class ModelDecompressor:
    """
    Decompress an entropy-coded model.
    """
    
    def load(self, path: Path) -> CompressedModel:
        """Load compressed model from file."""
        with open(path, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Invalid magic: {magic}")
            
            version = struct.unpack('<I', f.read(4))[0]
            if version != VERSION:
                raise ValueError(f"Unsupported version: {version}")
            
            n_tensors = struct.unpack('<I', f.read(4))[0]
            metadata_offset = struct.unpack('<Q', f.read(8))[0]
            data_offset = struct.unpack('<Q', f.read(8))[0]
            
            # Read metadata
            f.seek(metadata_offset)
            metadata_size = data_offset - metadata_offset
            metadata_json = f.read(metadata_size).decode('utf-8')
            metadata_dict = json.loads(metadata_json)
            
            # Read data
            f.seek(data_offset)
            data = f.read()
        
        # Reconstruct tensors
        tensors = {}
        for name, t in metadata_dict.items():
            tensors[name] = TensorMetadata(
                name=name,
                shape=t['shape'],
                dtype=t['dtype'],
                bits=t['bits'],
                scale=t['scale'],
                zero_point=t['zero_point'],
                data_offset=t['data_offset'],
                data_size=t['data_size'],
                table_offset=t['table_offset'],
                table_size=t['table_size'],
                n_elements=t['n_elements'],
                entropy=t['entropy'],
                compression_ratio=t['compression_ratio'],
            )
        
        return CompressedModel(tensors=tensors, data=data)
    
    def decompress_tensor(
        self,
        model: CompressedModel,
        name: str,
    ) -> np.ndarray:
        """
        Decompress a single tensor.
        
        Args:
            model: Loaded CompressedModel
            name: Tensor name
            
        Returns:
            Reconstructed float32 tensor
        """
        meta = model.tensors[name]
        
        # Extract table and data
        table_bytes = model.data[meta.table_offset:meta.table_offset + meta.table_size]
        data_bytes = model.data[meta.data_offset:meta.data_offset + meta.data_size]
        
        # Rebuild table
        table = RANSTable.from_bytes(table_bytes)
        
        # Decode
        tile = EncodedTile(
            data=data_bytes,
            n_symbols=meta.n_elements,
            table=table,
            original_bytes=(meta.n_elements * meta.bits + 7) // 8
        )
        indices = decode_tile(tile)
        
        # Dequantize
        weights = indices.astype(np.float32) * meta.scale + meta.zero_point
        
        return weights.reshape(meta.shape)
