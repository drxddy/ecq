#!/usr/bin/env python3
"""
Real Model Inference Test: Entropy-Coded Quantization

Tests entropy coding on SmolLM-135M with real weights.
"""

import time
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

from mlx.nn.layers.entropy_coded import EntropyCodedLinear, DecodeMode


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def __call__(self, x):
        B, L, D = x.shape
        
        # Get projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Repeat KV heads
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)
        
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        weights = mx.softmax(scores, axis=-1)
        output = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, eps: float):
        super().__init__()
        self.self_attn = Attention(dim, n_heads, n_kv_heads)
        self.mlp = MLP(dim, hidden_dim)
        self.input_layernorm = RMSNorm(dim, eps)
        self.post_attention_layernorm = RMSNorm(dim, eps)

    def __call__(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class SmolLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [
            TransformerBlock(
                config["hidden_size"],
                config["num_attention_heads"],
                config["num_key_value_heads"],
                config["intermediate_size"],
                config["rms_norm_eps"]
            )
            for _ in range(config["num_hidden_layers"])
        ]
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def __call__(self, x):
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


def load_weights(model, weights_path: Path):
    """Load safetensors weights into model."""
    weights = mx.load(str(weights_path))
    
    # Map HF keys to our model structure
    new_weights = {}
    for k, v in weights.items():
        # model.embed_tokens.weight -> embed_tokens.weight
        new_key = k.replace("model.", "")
        new_weights[new_key] = v
    
    # Handle tied embeddings: lm_head shares weight with embed_tokens
    if "lm_head.weight" not in new_weights and "embed_tokens.weight" in new_weights:
        new_weights["lm_head.weight"] = new_weights["embed_tokens.weight"]
    
    model.load_weights(list(new_weights.items()))
    return model


def convert_linear_to_entropy(model, decode_mode="fused_v2", n_streams=64, max_layers=None, group_size=64):
    """Convert Linear layers to EntropyCodedLinear with per-group quantization."""
    converted = 0
    ratios = []
    
    # Convert lm_head (large vocab - use larger group size)
    if isinstance(model.lm_head, nn.Linear):
        ec = EntropyCodedLinear.from_linear(
            model.lm_head, n_streams=n_streams, decode_mode=decode_mode, 
            group_size=group_size
        )
        model.lm_head = ec
        ratios.append(ec.compression_ratio)
        converted += 1
        print(f"      lm_head: {ec.compression_ratio:.2f}x compression, {ec.bits_per_weight:.2f} bits/weight")
    
    # Convert transformer layers
    layers_to_convert = model.layers[:max_layers] if max_layers else model.layers
    for i, layer in enumerate(layers_to_convert):
        # MLP
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            lin = getattr(layer.mlp, name, None)
            if isinstance(lin, nn.Linear):
                ec = EntropyCodedLinear.from_linear(
                    lin, n_streams=n_streams, decode_mode=decode_mode,
                    group_size=group_size
                )
                setattr(layer.mlp, name, ec)
                ratios.append(ec.compression_ratio)
                converted += 1
        
        # Attention
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            lin = getattr(layer.self_attn, name, None)
            if isinstance(lin, nn.Linear):
                ec = EntropyCodedLinear.from_linear(
                    lin, n_streams=n_streams, decode_mode=decode_mode,
                    group_size=group_size
                )
                setattr(layer.self_attn, name, ec)
                ratios.append(ec.compression_ratio)
                converted += 1
        
        if (i + 1) % 10 == 0:
            print(f"      Converted layer {i + 1}")
    
    return converted, ratios


def benchmark(model, input_ids, n_iters=10, warmup=3):
    """Benchmark forward pass."""
    for _ in range(warmup):
        out = model(input_ids)
        mx.eval(out)
    
    mx.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        out = model(input_ids)
        mx.eval(out)
    mx.synchronize()
    
    return (time.perf_counter() - start) / n_iters * 1000


def main():
    print("=" * 70)
    print("Real Model Inference: SmolLM-135M with Entropy Coding")
    print("=" * 70)
    
    model_path = Path("models/SmolLM-135M")
    config_path = model_path / "config.json"
    weights_path = model_path / "model.safetensors"
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\nModel: SmolLM-135M")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Layers: {config['num_hidden_layers']}")
    print(f"  Heads: {config['num_attention_heads']}")
    
    # Test with subset of layers for speed and memory
    max_layers = 2  # Use first 2 layers for faster testing
    print(f"\n  Testing with first {max_layers} layers for speed")
    
    # Create and load baseline model
    print("\n1. Loading baseline model...")
    model_base = SmolLM(config)
    model_base = load_weights(model_base, weights_path)  # Load full weights first
    model_base.layers = model_base.layers[:max_layers]  # Then subset
    mx.eval(model_base.parameters())
    print("   Loaded weights successfully")
    
    # Input - single token for autoregressive generation (GEMV case)
    # This is the real use case for entropy coding
    input_ids = mx.array([[100]])  # Single token
    
    # Baseline benchmark
    print("\n2. Benchmarking baseline (bfloat16)...")
    base_time = benchmark(model_base, input_ids)
    print(f"   Time: {base_time:.2f} ms")
    
    # Create FUSED_V2 model
    print("\n3. Creating entropy-coded model (FUSED_V2)...")
    model_v2 = SmolLM(config)
    model_v2 = load_weights(model_v2, weights_path)
    model_v2.layers = model_v2.layers[:max_layers]
    mx.eval(model_v2.parameters())
    
    # Use group_size=64 for per-group quantization (better compression)
    converted, ratios = convert_linear_to_entropy(model_v2, "fused_v2", max_layers=max_layers, group_size=64)
    print(f"   Converted {converted} layers")
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"   Avg compression: {avg_ratio:.2f}x over 4-bit")
        print(f"   Effective bits: {4.0 / avg_ratio:.2f} bits/weight")
    
    print("\n   Benchmarking FUSED_V2...")
    v2_time = benchmark(model_v2, input_ids)
    print(f"   Time: {v2_time:.2f} ms")
    print(f"   Overhead: {v2_time / base_time:.2f}x")
    
    # Create CACHED model  
    print("\n4. Creating entropy-coded model (CACHED)...")
    model_cached = SmolLM(config)
    model_cached = load_weights(model_cached, weights_path)
    model_cached.layers = model_cached.layers[:max_layers]
    mx.eval(model_cached.parameters())
    
    convert_linear_to_entropy(model_cached, "cached", max_layers=max_layers, group_size=64)
    
    print("\n   Benchmarking CACHED...")
    cached_time = benchmark(model_cached, input_ids)
    print(f"   Time: {cached_time:.2f} ms")
    print(f"   Overhead: {cached_time / base_time:.2f}x")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (SmolLM-135M, first 6 layers)")
    print("=" * 70)
    print(f"{'Mode':<20} {'Time (ms)':<12} {'vs Base':<12} {'Memory':<25}")
    print("-" * 70)
    print(f"{'Baseline (bf16)':<20} {base_time:<12.2f} {'1.00x':<12} {'100%':<25}")
    print(f"{'FUSED_V2':<20} {v2_time:<12.2f} {v2_time/base_time:<12.2f}x {f'{4.0/avg_ratio:.1f} bits/weight':<25}")
    print(f"{'CACHED':<20} {cached_time:<12.2f} {cached_time/base_time:<12.2f}x {'disk savings only':<25}")
    
    print(f"""
Conclusion:
- Real LLM weights achieve {avg_ratio:.2f}x compression over 4-bit
- Effective storage: {4.0/avg_ratio:.1f} bits/weight (vs 16 bits bf16)
- FUSED_V2 overhead: {v2_time/base_time:.1f}x (decode in kernel)
- CACHED overhead: {cached_time/base_time:.1f}x (pre-decoded)
- Total memory savings: ~{100 - 100*4.0/avg_ratio/16:.0f}% vs bf16
""")


if __name__ == "__main__":
    main()
