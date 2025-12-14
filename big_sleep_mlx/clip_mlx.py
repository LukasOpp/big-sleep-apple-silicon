"""
MLX-based CLIP implementation for Apple Silicon
Simplified version leveraging MLX's efficient operations
"""
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import numpy as np
import json
from typing import List, Union, Tuple


class LayerNorm(nn.Module):
    """MLX LayerNorm implementation"""
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x):
        means = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x_norm = (x - means) / mx.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class MultiHeadAttention(nn.Module):
    """MLX Multi-head attention"""
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.dims = dims
        self.head_dim = dims // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x, mask=None):
        B, L, D = x.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        
        # Combine heads
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block for Vision Transformer"""
    def __init__(self, dims: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = LayerNorm(dims)
        self.attn = MultiHeadAttention(dims, num_heads)
        self.ln2 = LayerNorm(dims)
        
        mlp_dims = int(dims * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_dims),
            nn.GELU(),
            nn.Linear(mlp_dims, dims)
        )

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for CLIP"""
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 32,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        output_dim: int = 512
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        
        # Patch embedding
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = width ** -0.5
        self.class_embedding = mx.random.normal((width,)) * scale
        num_patches = (image_size // patch_size) ** 2
        self.positional_embedding = mx.random.normal((num_patches + 1, width)) * scale
        
        self.ln_pre = LayerNorm(width)
        self.transformer = nn.Sequential(*[
            TransformerBlock(width, heads) for _ in range(layers)
        ])
        self.ln_post = LayerNorm(width)
        self.proj = mx.random.normal((width, output_dim)) * scale

    def __call__(self, x):
        # x shape: (B, H, W, C) in MLX format
        x = self.conv1(x)  # (B, H', W', width)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        
        # Add class token
        class_token = mx.broadcast_to(self.class_embedding, (B, 1, self.width))
        x = mx.concatenate([class_token, x], axis=1)
        
        # Add positional embedding
        x = x + self.positional_embedding
        
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj
        
        return x


class TextTransformer(nn.Module):
    """Text Transformer for CLIP"""
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        output_dim: int = 512
    ):
        super().__init__()
        self.context_length = context_length
        self.width = width
        
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = mx.random.normal((context_length, width))
        
        self.transformer = nn.Sequential(*[
            TransformerBlock(width, heads) for _ in range(layers)
        ])
        
        self.ln_final = LayerNorm(width)
        self.text_projection = mx.random.normal((width, output_dim))

    def __call__(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)
        
        # Take features from the EOT token
        x = x[mx.arange(x.shape[0]), text.argmax(axis=-1)] @ self.text_projection
        return x


class CLIPMLX(nn.Module):
    """MLX-based CLIP model"""
    def __init__(
        self,
        embed_dim: int = 512,
        image_resolution: int = 224,
        vision_layers: int = 12,
        vision_width: int = 768,
        vision_patch_size: int = 32,
        context_length: int = 77,
        vocab_size: int = 49408,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12
    ):
        super().__init__()
        
        self.context_length = context_length
        
        self.visual = VisionTransformer(
            image_size=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim
        )
        
        self.text = TextTransformer(
            context_length=context_length,
            vocab_size=vocab_size,
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            output_dim=embed_dim
        )
        
        self.logit_scale = mx.array([np.log(1 / 0.07)])

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        return self.text(text)

    def __call__(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Normalize features
        image_features = image_features / mx.sqrt(mx.sum(image_features**2, axis=-1, keepdims=True))
        text_features = text_features / mx.sqrt(mx.sum(text_features**2, axis=-1, keepdims=True))
        
        # Cosine similarity as logits
        logit_scale = mx.exp(self.logit_scale)
        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text


def load_clip_mlx(name: str = "ViT-B/32"):
    """
    Load CLIP model in MLX format
    For now, returns a basic architecture
    TODO: Add weight loading from converted PyTorch checkpoints
    """
    if name == "ViT-B/32":
        model = CLIPMLX(
            embed_dim=512,
            image_resolution=224,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=32,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12
        )
    elif name == "ViT-L/14":
        model = CLIPMLX(
            embed_dim=768,
            image_resolution=224,
            vision_layers=24,
            vision_width=1024,
            vision_patch_size=14,
            context_length=77,
            vocab_size=49408,
            transformer_width=768,
            transformer_heads=12,
            transformer_layers=12
        )
    else:
        raise ValueError(f"Model {name} not found")
    
    return model


# Simple tokenizer (simplified version)
def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> mx.array:
    """
    Simplified tokenizer for CLIP
    Returns: MLX array of token indices
    """
    if isinstance(texts, str):
        texts = [texts]
    
    # Simple character-level tokenization (placeholder)
    # TODO: Implement proper BPE tokenizer
    result = mx.zeros((len(texts), context_length), dtype=mx.int32)
    
    for i, text in enumerate(texts):
        tokens = [ord(c) % 49408 for c in text[:context_length]]
        result[i, :len(tokens)] = mx.array(tokens)
    
    return result
