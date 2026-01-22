# Strategies to Reduce/Eliminate Decode Overhead

## Current Situation

```
Per-token: [Load compressed] → [Decode] → [Dequantize] → [GEMV]
                  ↓               ↓
            Memory bound    24% overhead
```

**Problem:** We decode the same weights every token, wasting 24% of our bandwidth gain.

---

## Strategy 1: Decode-Once at Load Time

**Idea:** Decompress from disk → 4-bit in memory. Never decode again.

```
Load time:  [Disk: 1.1GB compressed] → [Decode] → [RAM: 3.5GB 4-bit]
Inference:  [Load 4-bit] → [Dequantize] → [GEMV]  (no decode!)
```

**Benefits:**
- 0% decode overhead during inference
- 3.2x faster model loading (read less from disk)
- Uses existing 4-bit kernels

**Tradeoffs:**
- Memory footprint = 4-bit (not compressed)
- Lose 1.84x memory bandwidth benefit

**When to use:** When RAM is plentiful, disk is slow, inference speed matters most.

---

## Strategy 2: Async Pre-decode (Double Buffering)

**Idea:** While GPU computes layer N, decode layer N+1 in parallel.

```
Timeline:
  GPU:  [Compute L0] [Compute L1] [Compute L2] ...
  CPU:  [Decode L1]  [Decode L2]  [Decode L3]  ...
              ↓           ↓           ↓
         Ready before GPU needs it!
```

**Implementation:**
```python
class AsyncDecoder:
    def __init__(self, model):
        self.decode_buffer = [None, None]  # Double buffer
        self.current = 0
        
    async def prefetch_layer(self, layer_idx):
        """Decode next layer while GPU is busy."""
        compressed = self.model.layers[layer_idx].compressed_weights
        decoded = entropy_decode(compressed)  # CPU work
        self.decode_buffer[1 - self.current] = decoded
        
    def get_decoded_weights(self, layer_idx):
        """Get pre-decoded weights, swap buffers."""
        weights = self.decode_buffer[self.current]
        self.current = 1 - self.current
        return weights
```

**Benefits:**
- 0% effective overhead (decode hidden by compute)
- Keeps compressed format in memory (1.84x smaller)
- Works with existing GPU kernels

**Tradeoffs:**
- Need 2 layers worth of 4-bit memory (~14MB for 7B)
- CPU must keep up with GPU (usually fine)
- Added complexity

---

## Strategy 3: GPU Async Decode (Metal Async Compute) ✅ IMPLEMENTED

**Idea:** Use Metal's async compute to decode on GPU while GEMV runs.

```metal
// Dispatch decode kernel on async queue
// Dispatch GEMV on main queue
// GPU scheduler interleaves them
```

**Metal Implementation:**
```cpp
// Create async compute queue
id<MTLCommandQueue> decodeQueue = [device newCommandQueue];
id<MTLCommandQueue> computeQueue = [device newCommandQueue];

// Layer N: GEMV on compute queue
[computeEncoder dispatchThreadgroups:...];

// Layer N+1: Decode on decode queue (runs in parallel!)
[decodeEncoder dispatchThreadgroups:...];
```

**Benefits:**
- Uses GPU's parallel execution capability
- No CPU involvement
- Decode "free" if GPU has spare compute

**Tradeoffs:**
- Requires careful synchronization
- GPU may not have spare cycles if fully utilized

---

## Strategy 4: Hybrid Cached Decode

**Idea:** Decode to a persistent cache, evict LRU layers.

```
First access:  [Compressed] → [Decode] → [Cache] → [Use]
Subsequent:    [Cache hit!] → [Use]  (no decode)

Cache eviction when memory pressure (keep hot layers)
```

**Implementation:**
```python
class DecodedWeightCache:
    def __init__(self, max_memory_gb=2.0):
        self.cache = {}
        self.max_bytes = max_memory_gb * 1e9
        self.access_order = []
        
    def get(self, layer_id, compressed_weights):
        if layer_id in self.cache:
            # Cache hit - move to front
            self.access_order.remove(layer_id)
            self.access_order.append(layer_id)
            return self.cache[layer_id]
        
        # Cache miss - decode
        decoded = entropy_decode(compressed_weights)
        
        # Evict if needed
        while self._current_size() + decoded.nbytes > self.max_bytes:
            evict_id = self.access_order.pop(0)
            del self.cache[evict_id]
        
        self.cache[layer_id] = decoded
        self.access_order.append(layer_id)
        return decoded
```

**Benefits:**
- Amortizes decode cost over multiple tokens
- Memory-adaptive (works within constraints)
- After warmup, approaches 0% overhead

**Tradeoffs:**
- First-token latency includes decode
- Memory overhead for cache

---

## Strategy 5: Simplified Codec (Trade Compression for Speed)

**Idea:** Use a faster codec that's easier to decode.

| Codec | Compression | Decode Speed | Complexity |
|-------|-------------|--------------|------------|
| rANS | 1.84x | Medium | Sequential state |
| Huffman | 1.80x | Fast | Table lookup |
| Fixed-width | 1.00x | Instant | None |
| **Block-ANS** | **1.75x** | **Fast** | **Parallel** |

**Block-ANS approach:**
- Divide weights into 32-element blocks
- Each block has independent rANS state
- Decode all blocks in parallel

```metal
// Each thread decodes one 32-element block
kernel void block_ans_decode(
    device uint8_t* compressed,
    device float* output,
    uint tid [[thread_position_in_grid]]
) {
    uint block_id = tid;
    
    // Each block: 4-byte state + ~16 bytes data
    // Fully parallel decode!
}
```

**Benefits:**
- Massive parallelism (1M+ blocks for 7B)
- No sequential dependencies
- Could approach 0% overhead

**Tradeoffs:**
- Slightly worse compression (block overhead)
- More complex encoder

---

## Recommended Approach: Hybrid Strategy

Combine multiple strategies based on context:

```python
class SmartDecoder:
    def __init__(self, model, strategy='auto'):
        self.strategy = strategy
        self.cache = DecodedWeightCache()
        self.prefetcher = AsyncPrefetcher()
        
    def get_weights(self, layer_idx):
        if self.strategy == 'decode_once':
            # Already decoded at load time
            return self.model.decoded_weights[layer_idx]
            
        elif self.strategy == 'prefetch':
            # Async pre-decoded
            self.prefetcher.start_next(layer_idx + 1)
            return self.prefetcher.get(layer_idx)
            
        elif self.strategy == 'cached':
            # LRU cache
            return self.cache.get(layer_idx, 
                                   self.model.compressed[layer_idx])
                                   
        else:  # 'fused'
            # Decode in GEMV kernel (current approach)
            return None  # Kernel handles it
```

---

## Implementation Priority

| Strategy | Complexity | Benefit | Priority |
|----------|------------|---------|----------|
| Decode-once at load | Low | Eliminates overhead | **1** |
| Block-ANS codec | Medium | Parallel decode | **2** |
| Async prefetch | Medium | Hides latency | 3 |
| LRU cache | Low | Amortizes decode | 4 |
| GPU async compute | High | Uses spare cycles | 5 |

---

## Quick Win: Decode-Once Mode

Add to MLX integration:

```python
class EntropyCodedLinear(nn.Module):
    def __init__(self, ..., decode_mode='fused'):
        """
        decode_mode:
          - 'fused': Decode in GEMV kernel (smallest memory)
          - 'cached': Decode once, keep in memory (fastest inference)
        """
        self.decode_mode = decode_mode
        
        if decode_mode == 'cached':
            # Decode at init time
            self.decoded_weights = self._decode_all()
            self.compressed_weights = None  # Free memory
            
    def forward(self, x):
        if self.decode_mode == 'cached':
            # Use pre-decoded weights with standard 4-bit kernel
            return quantized_matmul(x, self.decoded_weights, ...)
        else:
            # Use fused decode+GEMV kernel
            return entropy_coded_matmul(x, self.compressed_weights, ...)
```

This gives users a choice:
- **Memory-constrained:** Use `fused` mode (2.48x speedup, 1.84x smaller)
- **Speed-constrained:** Use `cached` mode (same as 4-bit speed, 3.2x faster load)
