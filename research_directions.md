# Research Directions: ALU-Rich, Bandwidth-Poor Inference

**Core Insight**: Apple Silicon has ~50 FLOPS per byte of bandwidth. 
We're using ~2 FLOPS per byte. There's 25x headroom to trade compute for memory.

---

## 1. Implicit Neural Representations for Weights (INR-Weights)

**Idea**: Instead of storing weights, store a tiny MLP that *generates* weights.

```
Traditional: Load W[i,j] from memory
INR:         W[i,j] = MLP(i, j; θ)  where |θ| << |W|
```

**Math**:
```
W ∈ ℝ^(4096 × 4096) = 64MB (fp16)

Replace with:
  θ = 3-layer MLP with 256 hidden units
  |θ| ≈ 256KB (400x compression)
  
  W[i,j] = σ(σ(σ([i,j] @ W1) @ W2) @ W3)
```

**Why it might work**:
- Weight matrices have structure (low-rank, smooth regions)
- NeRF proved MLPs can memorize complex functions
- Apple's Neural Engine could accelerate the inner MLP

**Research needed**:
- Can a small MLP memorize a weight matrix with low error?
- What's the compute/quality tradeoff?

---

## 2. Frequency-Domain Weight Compression (DCT-Quant)

**Idea**: Weights have spatial structure. Transform to frequency domain, keep only low frequencies.

```
W_orig ∈ ℝ^(n × m)
W_dct = DCT2D(W_orig)         # Transform
W_sparse = TopK(W_dct, k)     # Keep k largest coefficients
W_recon = IDCT2D(W_sparse)    # Reconstruct via compute
```

**Math**:
```
For a 4096×4096 matrix:
  - Original: 16M values
  - If 95% energy in 5% coefficients: 800K values
  - Compression: 20x
  
DCT basis functions are computed, not stored:
  DCT[i,j] = cos(π(2i+1)j / 2N)  # Pure ALU
```

**Why it might work**:
- JPEG uses this successfully for images
- Weight matrices often have smooth regions
- DCT/IDCT are pure compute, no memory access

---

## 3. Learned Vector Quantization (LVQ) with Tiny Codebooks

**Idea**: Group weights into vectors, quantize to learned centroids, reconstruct via small matmul.

```
W ∈ ℝ^(4096 × 4096)
Reshape to: V ∈ ℝ^(1M × 16)  # 1M vectors of dim 16

Codebook: C ∈ ℝ^(256 × 16)   # 256 learned centroids
Indices:  I ∈ [0,255]^1M     # 1 byte per vector = 1MB

Reconstruct: W_recon = C[I].reshape(4096, 4096)
```

**Math**:
```
Original:     64 MB
Codebook:     8 KB (fits in L1 cache!)
Indices:      1 MB  
Total:        ~1 MB (64x compression)

Reconstruction: Just indexing + reshape
  - Codebook in registers
  - Sequential index access
  - Pure cache hits
```

**This is essentially what AQLM/QuIP# do** — worth implementing for MLX.

---

## 4. Weight Prediction via Recurrence

**Idea**: Weights have patterns. Predict next weight from previous weights.

```
W[i] = f(W[i-1], W[i-2], ..., W[i-k]; θ)

Store: W[0:k] (seed) + θ (predictor params)
Generate: Remaining weights via recurrence
```

**Math**:
```
Linear predictor: W[i] = Σ α_j × W[i-j]
  - k=8 coefficients: 16 bytes
  - Generates unlimited weights
  
Error correction:
  Store residuals only where |W_pred - W_true| > threshold
  Sparse residual storage
```

**Why it might work**:
- Adjacent weights in a row often correlated
- LPC (Linear Predictive Coding) works for audio
- Could achieve very high compression for smooth weight regions

---

## 5. Sparse Factorization: W = UV + S

**Idea**: Low-rank approximation + sparse correction.

```
W ∈ ℝ^(4096 × 4096)

Decompose:
  U ∈ ℝ^(4096 × r)    # r = 64 (low rank)
  V ∈ ℝ^(r × 4096)
  S ∈ sparse          # <1% non-zero

W ≈ U @ V + S
```

**Math**:
```
Original:        64 MB
U (rank-64):     512 KB
V (rank-64):     512 KB  
S (1% sparse):   ~1.3 MB (index + value)
Total:           ~2.3 MB (28x compression)

Compute: 
  y = (U @ V) @ x + S @ x
  = U @ (V @ x) + S @ x   # Associative reorder
  
  Cost: 2 × (4096 × 64) + sparse = ~500K FLOPs
  vs loading 64MB at 150GB/s = 0.4ms
  
  500K FLOPs at 7 TFLOPS = 0.07ms  ✓ Faster!
```

**This is promising** — compute cheaper than memory access.

---

## 6. Mixture-of-Experts with Aggressive Routing

**Idea**: Only load the expert weights you need.

```
Instead of: W ∈ ℝ^(4096 × 14336)  # Full MLP

Use: 8 experts × W_e ∈ ℝ^(4096 × 1792)  # Each 1/8 size
     Router selects top-2 experts

Memory loaded: 2/8 = 25% of weights
```

**For Apple Silicon**:
- Unified memory means expert switching is fast
- Could have 32 experts, activate 2 → 94% memory reduction
- Matches Mixtral architecture

---

## 7. Compute-in-Superposition

**Idea**: Store weights in a compressed "superposition" and extract via compute.

```
W1, W2, ..., Wk are k weight matrices
Store: M = Σ φ_i ⊗ W_i  (superposition encoding)

Retrieve W_j: W_j ≈ <φ_j, M>  (inner product extraction)
```

**Math** (based on Hopfield networks / superposition):
```
If φ_i are random orthogonal vectors:
  <φ_j, M> = <φ_j, Σ φ_i ⊗ W_i>
           = W_j + noise  (cross-talk from other W_i)

With k matrices, noise ∝ √k
Capacity: Can store O(d) matrices in d-dimensional space
```

**Speculative but interesting** — could store multiple layers in one tensor.

---

## 8. Algorithmic Weight Generation (HashNet)

**Idea**: Generate weights deterministically from a hash function.

```
W[i,j] = Hash(layer_id, i, j, seed) → float

No storage needed — pure compute!
```

**Research** (from "Compressing Neural Networks with the Hashing Trick"):
- Random weights work surprisingly well for some layers
- Hash collisions = weight sharing = regularization
- Could work for less critical layers

---

## Recommended Priority

| Idea | Compression | Compute Cost | Implementation Difficulty | Potential |
|------|-------------|--------------|---------------------------|-----------|
| **5. Sparse Factorization** | 20-50x | Medium | Medium | ⭐⭐⭐⭐⭐ |
| **3. Vector Quantization** | 30-60x | Low | Medium | ⭐⭐⭐⭐ |
| **2. DCT Compression** | 10-20x | Medium | Low | ⭐⭐⭐ |
| **6. MoE Routing** | 4-8x | Low | High | ⭐⭐⭐ |
| **1. INR Weights** | 100x+ | High | High | ⭐⭐ (research) |

---

## Next Step Recommendation

**Start with Sparse Factorization (W = UV + S)**:
1. Mathematically sound (SVD-based)
2. Proven compute < memory on Apple Silicon
3. Can implement in pure Python first
4. Directly reduces memory bandwidth

Want me to prototype this?
