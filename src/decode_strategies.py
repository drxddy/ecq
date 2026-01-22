#!/usr/bin/env python3
"""
Decode Strategies for Entropy-Coded Quantization

Different strategies to handle decode overhead:
1. DecodeOnce: Decode at load time, store 4-bit (fastest inference)
2. Fused: Decode in GEMV kernel (smallest memory)
3. Cached: LRU cache of decoded layers (balanced)
4. Prefetch: Async decode of next layer (hides latency)
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import threading
from collections import OrderedDict

from .interleaved_rans import InterleavedEncodedTile, interleaved_rans_decode


class DecodeMode(Enum):
    FUSED = "fused"          # Decode in kernel (current approach)
    CACHED = "cached"        # Decode once at load, keep in memory
    LRU = "lru"              # LRU cache of decoded layers
    PREFETCH = "prefetch"    # Async prefetch next layer (CPU threading)
    GPU_ASYNC = "gpu_async"  # Async GPU decode with Metal command queues


@dataclass
class DecodedWeights:
    """Container for decoded 4-bit weights."""
    indices: np.ndarray  # uint8, 4-bit quantized values
    scale: float
    zero_point: float
    shape: tuple
    
    @property
    def nbytes(self) -> int:
        return self.indices.nbytes


class DecodeOnceStrategy:
    """
    Decode at model load time, store as 4-bit.
    
    Tradeoff: Uses 4-bit memory footprint, but 0% decode overhead.
    Best for: Speed-critical inference with sufficient RAM.
    """
    
    def __init__(self):
        self.decoded_layers: Dict[str, DecodedWeights] = {}
        
    def load_layer(self, name: str, tile: InterleavedEncodedTile, 
                   scale: float, zero_point: float, shape: tuple) -> None:
        """Decode and store a layer at load time."""
        # Decode using CPU (one-time cost)
        indices = interleaved_rans_decode(tile)
        
        self.decoded_layers[name] = DecodedWeights(
            indices=indices.reshape(shape),
            scale=scale,
            zero_point=zero_point,
            shape=shape
        )
        
    def get_weights(self, name: str) -> DecodedWeights:
        """Get pre-decoded weights (instant, no decode)."""
        return self.decoded_layers[name]
    
    def memory_usage(self) -> int:
        """Total memory used by decoded weights."""
        return sum(w.nbytes for w in self.decoded_layers.values())


class LRUCacheStrategy:
    """
    LRU cache of decoded layers.
    
    Tradeoff: First access decodes, subsequent accesses are instant.
    Best for: Repeated inference with memory constraints.
    """
    
    def __init__(self, max_memory_bytes: int = 2 * 1024**3):  # 2GB default
        self.max_bytes = max_memory_bytes
        self.cache: OrderedDict[str, DecodedWeights] = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        
    def get_weights(self, name: str, tile: InterleavedEncodedTile,
                    scale: float, zero_point: float, shape: tuple) -> DecodedWeights:
        """Get weights, decoding if not in cache."""
        
        if name in self.cache:
            # Cache hit - move to end (most recently used)
            self.cache.move_to_end(name)
            self.hits += 1
            return self.cache[name]
        
        # Cache miss - decode
        self.misses += 1
        indices = interleaved_rans_decode(tile)
        
        decoded = DecodedWeights(
            indices=indices.reshape(shape),
            scale=scale,
            zero_point=zero_point,
            shape=shape
        )
        
        # Evict if needed
        while self.current_bytes + decoded.nbytes > self.max_bytes and self.cache:
            evicted_name, evicted = self.cache.popitem(last=False)
            self.current_bytes -= evicted.nbytes
        
        # Add to cache
        self.cache[name] = decoded
        self.current_bytes += decoded.nbytes
        
        return decoded
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PrefetchStrategy:
    """
    Async prefetch of next layer while current layer computes.
    
    Tradeoff: Uses 2 layers of memory, but hides decode latency.
    Best for: Sequential layer processing with predictable access.
    """
    
    def __init__(self):
        self.buffers: Dict[str, DecodedWeights] = {}
        self.pending: Dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        
    def prefetch(self, name: str, tile: InterleavedEncodedTile,
                 scale: float, zero_point: float, shape: tuple) -> None:
        """Start async decode of a layer."""
        
        def decode_worker():
            indices = interleaved_rans_decode(tile)
            decoded = DecodedWeights(
                indices=indices.reshape(shape),
                scale=scale,
                zero_point=zero_point,
                shape=shape
            )
            with self.lock:
                self.buffers[name] = decoded
        
        thread = threading.Thread(target=decode_worker)
        thread.start()
        self.pending[name] = thread
        
    def get_weights(self, name: str) -> DecodedWeights:
        """Get prefetched weights, waiting if still decoding."""
        
        if name in self.pending:
            self.pending[name].join()
            del self.pending[name]
        
        with self.lock:
            return self.buffers.pop(name)
    
    def clear(self, name: str) -> None:
        """Free memory after layer is done."""
        with self.lock:
            if name in self.buffers:
                del self.buffers[name]


class SmartDecoder:
    """
    Unified interface for all decode strategies.
    
    Usage:
        decoder = SmartDecoder(mode='cached')
        
        # At load time (for 'cached' mode)
        decoder.load_layer('layer.0.weight', tile, scale, zp, shape)
        
        # At inference time
        weights = decoder.get_weights('layer.0.weight', tile, scale, zp, shape)
        
    Available modes:
        - 'fused': Decode in GEMV kernel (smallest memory, decode overhead per token)
        - 'cached': Decode once at load, keep in memory (fastest inference)
        - 'lru': LRU cache of decoded layers (balanced memory/speed)
        - 'prefetch': CPU async prefetch next layer (hides decode latency)
        - 'gpu_async': GPU async decode with Metal (best for Apple Silicon)
    """
    
    def __init__(self, mode: str = 'fused', **kwargs):
        self.mode = DecodeMode(mode)
        
        if self.mode == DecodeMode.CACHED:
            self.strategy = DecodeOnceStrategy()
        elif self.mode == DecodeMode.LRU:
            max_memory = kwargs.get('max_memory_bytes', 2 * 1024**3)
            self.strategy = LRUCacheStrategy(max_memory)
        elif self.mode == DecodeMode.PREFETCH:
            self.strategy = PrefetchStrategy()
        elif self.mode == DecodeMode.GPU_ASYNC:
            self.strategy = GPUAsyncDecodeStrategy()
        else:
            self.strategy = None  # Fused mode - no separate decode
            
    def load_layer(self, name: str, tile: InterleavedEncodedTile,
                   scale: float, zero_point: float, shape: tuple) -> None:
        """Pre-decode a layer (for 'cached' mode)."""
        if self.mode == DecodeMode.CACHED:
            self.strategy.load_layer(name, tile, scale, zero_point, shape)
            
    def prefetch_layer(self, name: str, tile: InterleavedEncodedTile,
                       scale: float, zero_point: float, shape: tuple) -> None:
        """Start async decode (for 'prefetch' and 'gpu_async' modes)."""
        if self.mode == DecodeMode.PREFETCH:
            self.strategy.prefetch(name, tile, scale, zero_point, shape)
        elif self.mode == DecodeMode.GPU_ASYNC:
            self.strategy.prefetch_async(name, tile, scale, zero_point, shape)
            
    def get_weights(self, name: str, tile: Optional[InterleavedEncodedTile] = None,
                    scale: float = 0, zero_point: float = 0, 
                    shape: tuple = ()) -> Optional[DecodedWeights]:
        """
        Get decoded weights.
        
        For 'fused' mode, returns None (kernel handles decode).
        For other modes, returns DecodedWeights.
        """
        if self.mode == DecodeMode.FUSED:
            return None
        elif self.mode == DecodeMode.CACHED:
            return self.strategy.get_weights(name)
        elif self.mode == DecodeMode.LRU:
            return self.strategy.get_weights(name, tile, scale, zero_point, shape)
        elif self.mode == DecodeMode.PREFETCH:
            return self.strategy.get_weights(name)
        elif self.mode == DecodeMode.GPU_ASYNC:
            return self.strategy.wait_and_get(name)
        return None
    
    def is_ready(self, name: str) -> bool:
        """Check if a layer is ready (for async modes)."""
        if self.mode == DecodeMode.GPU_ASYNC:
            return self.strategy.is_ready(name)
        elif self.mode == DecodeMode.PREFETCH:
            return name in self.strategy.buffers
        return True
    
    def memory_usage(self) -> int:
        """Current memory usage in bytes."""
        if self.mode == DecodeMode.CACHED:
            return self.strategy.memory_usage()
        elif self.mode == DecodeMode.LRU:
            return self.strategy.current_bytes
        elif self.mode == DecodeMode.GPU_ASYNC:
            return self.strategy.memory_usage()
        return 0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        if self.mode == DecodeMode.LRU:
            return {
                'hit_rate': self.strategy.hit_rate,
                'hits': self.strategy.hits,
                'misses': self.strategy.misses,
                'memory_bytes': self.strategy.current_bytes,
            }
        elif self.mode == DecodeMode.GPU_ASYNC:
            return {
                'metal_available': self.strategy.metal_available,
                'pending_decodes': len(self.strategy.pending_decodes),
                'decoded_buffers': len(self.strategy.decoded_buffers),
                'memory_bytes': self.strategy.memory_usage(),
            }
        return {}


class GPUAsyncDecodeStrategy:
    """
    GPU Async Decode: Decode layer N+1 while computing layer N.
    
    Uses Metal's concurrent command queues to overlap decode and compute.
    This hides decode latency when GPU has spare cycles.
    
    Memory requirement: 2 layers of decoded weights (~28MB for 7B)
    
    Architecture:
    - decode_queue: Dedicated queue for rANS decode operations
    - compute_queue: Main queue for GEMV/inference (external)
    - Uses MTLEvent for synchronization between queues
    """
    
    def __init__(self):
        self.metal_available = False
        self.device = None
        self.decode_queue = None
        self.library = None
        self.decode_pipeline = None
        
        try:
            import Metal
            self.Metal = Metal
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device is None:
                raise RuntimeError("No Metal device found")
            
            self.decode_queue = self.device.newCommandQueue()
            self._compile_shader()
            self.metal_available = True
        except (ImportError, RuntimeError) as e:
            print(f"Metal not available: {e}, falling back to CPU")
            
        self.decoded_buffers: Dict[str, DecodedWeights] = {}
        self.pending_decodes: Dict[str, Any] = {}  # layer_id -> (cmd_buffer, output_buf, metadata)
        self.lock = threading.Lock()
        
    def _compile_shader(self):
        """Compile the rANS decode shader."""
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        
        constant uint PROB_BITS = 14;
        constant uint PROB_SCALE = 1 << PROB_BITS;
        constant uint RANS_L = 1 << 23;
        
        kernel void parallel_rans_decode_async(
            device const uint8_t* compressed [[buffer(0)]],
            device const uint* stream_lengths [[buffer(1)]],
            device const uint16_t* freq [[buffer(2)]],
            device const uint16_t* cumfreq [[buffer(3)]],
            device const uint8_t* sym_table [[buffer(4)]],
            device uint8_t* output [[buffer(5)]],
            constant uint& n_streams [[buffer(6)]],
            constant uint& n_symbols [[buffer(7)]],
            constant uint& max_stream_len [[buffer(8)]],
            uint tid [[thread_position_in_grid]]
        ) {
            uint stream_idx = tid;
            if (stream_idx >= n_streams) return;
            
            // Cache tables in registers
            uint16_t local_freq[16];
            uint16_t local_cumfreq[16];
            for (int i = 0; i < 16; i++) {
                local_freq[i] = freq[i];
                local_cumfreq[i] = cumfreq[i];
            }
            
            uint stream_len = stream_lengths[stream_idx];
            if (stream_len < 4) return;
            
            // Coalesced reads for state initialization
            uint b0 = compressed[stream_idx + 0 * n_streams];
            uint b1 = compressed[stream_idx + 1 * n_streams];
            uint b2 = compressed[stream_idx + 2 * n_streams];
            uint b3 = compressed[stream_idx + 3 * n_streams];
            uint state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
            uint ptr = 4;
            
            uint symbols_in_stream = (n_symbols - stream_idx + n_streams - 1) / n_streams;
            
            for (uint i = 0; i < symbols_in_stream; i++) {
                uint output_idx = stream_idx + i * n_streams;
                if (output_idx >= n_symbols) break;
                
                uint slot = state & (PROB_SCALE - 1);
                uint8_t s = sym_table[slot];
                output[output_idx] = s;
                
                uint freq_s = local_freq[s];
                uint start_s = local_cumfreq[s];
                state = freq_s * (state >> PROB_BITS) + slot - start_s;
                
                while (state < RANS_L && ptr < stream_len) {
                    uint8_t b = compressed[stream_idx + ptr * n_streams];
                    state = (state << 8) | b;
                    ptr++;
                }
            }
        }
        """
        
        options = self.Metal.MTLCompileOptions.new()
        self.library, error = self.device.newLibraryWithSource_options_error_(
            shader_source, options, None
        )
        if error:
            raise RuntimeError(f"Shader compilation failed: {error}")
        
        decode_fn = self.library.newFunctionWithName_("parallel_rans_decode_async")
        self.decode_pipeline, error = self.device.newComputePipelineStateWithFunction_error_(
            decode_fn, None
        )
        if error:
            raise RuntimeError(f"Pipeline creation failed: {error}")
    
    def _make_buffer(self, data: np.ndarray) -> Any:
        """Create a Metal buffer from numpy array."""
        return self.device.newBufferWithBytes_length_options_(
            data.tobytes(),
            data.nbytes,
            self.Metal.MTLResourceStorageModeShared
        )
    
    def _make_buffer_empty(self, size: int) -> Any:
        """Create an empty Metal buffer."""
        return self.device.newBufferWithLength_options_(
            size,
            self.Metal.MTLResourceStorageModeShared
        )
        
    def prefetch_async(self, layer_id: str, tile: InterleavedEncodedTile, 
                       scale: float, zero_point: float, shape: tuple) -> None:
        """
        Start async GPU decode of a layer.
        
        The decode runs on a separate command queue, allowing it to execute
        concurrently with compute operations on the main queue.
        """
        if not self.metal_available:
            self._prefetch_cpu_fallback(layer_id, tile, scale, zero_point, shape)
            return
        
        # Prepare buffers
        compressed_buf = self._make_buffer(np.frombuffer(tile.data, dtype=np.uint8))
        lengths_buf = self._make_buffer(np.array(tile.stream_lengths, dtype=np.uint32))
        freq_buf = self._make_buffer(tile.table.freq.astype(np.uint16))
        cumfreq_buf = self._make_buffer(tile.table.cumfreq.astype(np.uint16))
        sym_table_buf = self._make_buffer(tile.table.sym_table.astype(np.uint8))
        output_buf = self._make_buffer_empty(tile.n_symbols)
        
        n_streams_buf = self._make_buffer(np.array([tile.n_streams], dtype=np.uint32))
        n_symbols_buf = self._make_buffer(np.array([tile.n_symbols], dtype=np.uint32))
        max_stream_len_buf = self._make_buffer(np.array([tile.max_stream_len], dtype=np.uint32))
        
        # Create and encode command buffer
        cmd_buffer = self.decode_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        
        encoder.setComputePipelineState_(self.decode_pipeline)
        encoder.setBuffer_offset_atIndex_(compressed_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(lengths_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(freq_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(cumfreq_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(sym_table_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(output_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(n_streams_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(n_symbols_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(max_stream_len_buf, 0, 8)
        
        # Dispatch one thread per stream
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.Metal.MTLSizeMake(tile.n_streams, 1, 1),
            self.Metal.MTLSizeMake(min(tile.n_streams, 256), 1, 1)
        )
        
        encoder.endEncoding()
        
        # Commit but DON'T wait - let it run async
        cmd_buffer.commit()
        
        # Store pending decode info
        with self.lock:
            self.pending_decodes[layer_id] = {
                'cmd_buffer': cmd_buffer,
                'output_buf': output_buf,
                'n_symbols': tile.n_symbols,
                'scale': scale,
                'zero_point': zero_point,
                'shape': shape,
                # Keep buffers alive until command completes
                'buffers': [compressed_buf, lengths_buf, freq_buf, cumfreq_buf, 
                           sym_table_buf, n_streams_buf, n_symbols_buf, max_stream_len_buf]
            }
    
    def _prefetch_cpu_fallback(self, layer_id: str, tile: InterleavedEncodedTile,
                               scale: float, zero_point: float, shape: tuple) -> None:
        """CPU fallback using background thread."""
        def decode_worker():
            indices = interleaved_rans_decode(tile)
            with self.lock:
                self.decoded_buffers[layer_id] = DecodedWeights(
                    indices=indices.reshape(shape),
                    scale=scale,
                    zero_point=zero_point,
                    shape=shape
                )
        
        thread = threading.Thread(target=decode_worker)
        thread.start()
        with self.lock:
            self.pending_decodes[layer_id] = {'thread': thread}
        
    def wait_and_get(self, layer_id: str) -> DecodedWeights:
        """Wait for async decode to complete and return weights."""
        with self.lock:
            # Already decoded?
            if layer_id in self.decoded_buffers:
                return self.decoded_buffers.pop(layer_id)
            
            pending = self.pending_decodes.get(layer_id)
            if pending is None:
                raise KeyError(f"Layer {layer_id} not found in pending or decoded")
        
        # CPU fallback path
        if 'thread' in pending:
            pending['thread'].join()
            with self.lock:
                del self.pending_decodes[layer_id]
                return self.decoded_buffers.pop(layer_id)
        
        # GPU path - wait for command buffer
        cmd_buffer = pending['cmd_buffer']
        cmd_buffer.waitUntilCompleted()
        
        # Read result from GPU buffer
        output_buf = pending['output_buf']
        n_symbols = pending['n_symbols']
        
        indices = np.frombuffer(
            output_buf.contents().as_buffer(n_symbols),
            dtype=np.uint8
        ).copy()
        
        decoded = DecodedWeights(
            indices=indices.reshape(pending['shape']),
            scale=pending['scale'],
            zero_point=pending['zero_point'],
            shape=pending['shape']
        )
        
        with self.lock:
            del self.pending_decodes[layer_id]
        
        return decoded
    
    def is_ready(self, layer_id: str) -> bool:
        """Check if decode is complete without blocking."""
        with self.lock:
            if layer_id in self.decoded_buffers:
                return True
            
            pending = self.pending_decodes.get(layer_id)
            if pending is None:
                return False
        
        # CPU fallback
        if 'thread' in pending:
            return not pending['thread'].is_alive()
        
        # GPU path - check command buffer status
        cmd_buffer = pending['cmd_buffer']
        # MTLCommandBufferStatus: 0=NotEnqueued, 1=Enqueued, 2=Committed, 3=Scheduled, 4=Completed, 5=Error
        return cmd_buffer.status() >= 4
    
    def get_if_ready(self, layer_id: str) -> Optional[DecodedWeights]:
        """Get decoded weights if ready, None otherwise (non-blocking)."""
        if self.is_ready(layer_id):
            return self.wait_and_get(layer_id)
        return None
    
    def cancel(self, layer_id: str) -> None:
        """Cancel a pending decode (best effort)."""
        with self.lock:
            if layer_id in self.pending_decodes:
                del self.pending_decodes[layer_id]
            if layer_id in self.decoded_buffers:
                del self.decoded_buffers[layer_id]
    
    def memory_usage(self) -> int:
        """Current memory used by decoded buffers."""
        with self.lock:
            return sum(w.nbytes for w in self.decoded_buffers.values())


def benchmark_strategies():
    """Benchmark different decode strategies."""
    import time
    from .rans import RANSTable
    from .interleaved_rans import interleaved_rans_encode
    
    print("="*60)
    print(" Decode Strategy Benchmark")
    print("="*60)
    
    # Create test data (simulating a layer)
    np.random.seed(42)
    n_elements = 1_000_000
    raw = np.random.randn(n_elements)
    indices = ((raw - raw.min()) / (raw.max() - raw.min()) * 15).astype(np.uint8)
    
    # Encode
    counts = np.bincount(indices, minlength=16)
    table = RANSTable.from_counts(counts)
    from .interleaved_rans import interleaved_rans_encode
    tile = interleaved_rans_encode(indices, table, n_streams=256)
    
    shape = (1000, 1000)
    scale = 0.1
    zp = -0.05
    
    results = {}
    
    # Strategy 1: Fused (baseline - decode every time)
    print("\n1. Fused (decode every access):")
    times = []
    for _ in range(10):
        start = time.perf_counter()
        decoded = interleaved_rans_decode(tile)
        times.append(time.perf_counter() - start)
    avg = np.mean(times) * 1000
    print(f"   Decode time: {avg:.2f} ms per access")
    results['fused'] = avg
    
    # Strategy 2: Cached (decode once)
    print("\n2. Cached (decode once at load):")
    decoder = SmartDecoder(mode='cached')
    
    start = time.perf_counter()
    decoder.load_layer('test', tile, scale, zp, shape)
    load_time = (time.perf_counter() - start) * 1000
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = decoder.get_weights('test')
        times.append(time.perf_counter() - start)
    access_time = np.mean(times) * 1000
    
    print(f"   Load time: {load_time:.2f} ms (one-time)")
    print(f"   Access time: {access_time:.4f} ms (per access)")
    print(f"   Memory: {decoder.memory_usage() / 1024**2:.1f} MB")
    results['cached'] = access_time
    
    # Strategy 3: LRU Cache
    print("\n3. LRU Cache (decode on miss):")
    decoder = SmartDecoder(mode='lru', max_memory_bytes=100*1024**2)
    
    # First access (cold)
    start = time.perf_counter()
    _ = decoder.get_weights('test', tile, scale, zp, shape)
    cold_time = (time.perf_counter() - start) * 1000
    
    # Subsequent accesses (warm)
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = decoder.get_weights('test', tile, scale, zp, shape)
        times.append(time.perf_counter() - start)
    warm_time = np.mean(times) * 1000
    
    print(f"   Cold access: {cold_time:.2f} ms")
    print(f"   Warm access: {warm_time:.4f} ms")
    print(f"   Hit rate: {decoder.stats['hit_rate']*100:.1f}%")
    results['lru_warm'] = warm_time
    
    # Strategy 4: GPU Async
    print("\n4. GPU Async (Metal command queue):")
    try:
        decoder = SmartDecoder(mode='gpu_async')
        stats = decoder.stats
        
        if stats.get('metal_available', False):
            # Test async prefetch + get
            start = time.perf_counter()
            decoder.prefetch_layer('test', tile, scale, zp, shape)
            prefetch_time = (time.perf_counter() - start) * 1000
            
            # Simulate some compute work
            _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 100)
            
            start = time.perf_counter()
            weights = decoder.get_weights('test', tile, scale, zp, shape)
            get_time = (time.perf_counter() - start) * 1000
            
            print(f"   Prefetch dispatch: {prefetch_time:.2f} ms")
            print(f"   Get (after compute): {get_time:.2f} ms")
            print(f"   Metal available: {stats['metal_available']}")
            
            # Benchmark raw GPU decode time
            times = []
            for i in range(10):
                start = time.perf_counter()
                decoder.prefetch_layer(f'bench_{i}', tile, scale, zp, shape)
                weights = decoder.get_weights(f'bench_{i}', tile, scale, zp, shape)
                times.append(time.perf_counter() - start)
            gpu_decode_time = np.mean(times) * 1000
            print(f"   GPU decode (sync): {gpu_decode_time:.2f} ms")
            results['gpu_async'] = gpu_decode_time
        else:
            print("   Metal not available, using CPU fallback")
            results['gpu_async'] = results['fused']
    except Exception as e:
        print(f"   GPU Async failed: {e}")
        results['gpu_async'] = results['fused']
    
    # Summary
    print("\n" + "="*60)
    print(" Summary: Access Time per Layer")
    print("="*60)
    print(f"   Fused (CPU):  {results['fused']:.2f} ms (decode every time)")
    print(f"   Cached:       {results['cached']:.4f} ms (pre-decoded)")
    print(f"   LRU:          {results['lru_warm']:.4f} ms (after warmup)")
    if 'gpu_async' in results:
        print(f"   GPU Async:    {results['gpu_async']:.2f} ms (Metal decode)")
    print()
    print(f"   Speedup (Cached vs Fused): {results['fused']/results['cached']:.0f}x")
    if 'gpu_async' in results and results['gpu_async'] > 0:
        print(f"   Speedup (GPU vs CPU):      {results['fused']/results['gpu_async']:.1f}x")


if __name__ == "__main__":
    benchmark_strategies()
