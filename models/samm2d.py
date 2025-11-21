"""
SAMM2D: Scale-Aware Multi-Modal 2D Transformer for Intracranial Aneurysm Detection
Author: Antara Das
Institution: Carnegie Mellon University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Converts 2D image patches into embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.position_embedding
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing TOF and MRA modalities.
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_modal, key_value_modal):
        """
        query_modal: features from one modality (e.g., TOF)
        key_value_modal: features from another modality (e.g., MRA)
        """
        B, N, C = query_modal.shape
        
        q = self.q_proj(query_modal).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value_modal).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value_modal).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-modal attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention for fusing multi-scale features.
    """
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_scale, key_value_scale):
        """
        query_scale: features from one scale (e.g., original)
        key_value_scale: features from another scale (e.g., downsampled)
        """
        B, N, C = query_scale.shape
        
        q = self.q_proj(query_scale).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value_scale).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value_scale).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-scale attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block.
    """
    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SAMM2D(nn.Module):
    """
    Scale-Aware Multi-Modal 2D Transformer for Intracranial Aneurysm Detection.
    
    Args:
        img_size: Input image size (default: 224)
        patch_size: Patch size for embedding (default: 16)
        in_channels: Number of input channels per modality (default: 1)
        num_classes: Number of output classes (default: 2 for binary)
        embed_dim: Embedding dimension (default: 256)
        depth: Number of transformer layers (default: 3)
        num_heads: Number of attention heads (default: 4)
        mlp_ratio: MLP hidden dimension ratio (default: 4)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=2,
        embed_dim=256,
        depth=3,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embeddings for each modality and scale
        self.tof_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.mra_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # For multi-scale (original, downsampled, upsampled)
        self.scale_orig_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.scale_down_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.scale_up_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Modal-specific transformer encoders
        self.modal_encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Scale-specific transformer encoders
        self.scale_encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Cross-modal attention
        self.cross_modal_attn = CrossModalAttention(embed_dim, num_heads, dropout)
        
        # Cross-scale attention
        self.cross_scale_attn = CrossScaleAttention(embed_dim, num_heads, dropout)
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim * 2),  # 6 = 2 modals * 3 cross-attn outputs for each
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, tof, mra, scale_orig, scale_down, scale_up):
        """
        Args:
            tof: TOF angiography images [B, 1, H, W]
            mra: MRA sequence images [B, 1, H, W]
            scale_orig: Original scale images [B, 1, H, W]
            scale_down: Downsampled scale images [B, 1, H, W]
            scale_up: Upsampled scale images [B, 1, H, W]
        
        Returns:
            logits: Classification logits [B, num_classes]
        """
        # Patch embedding
        tof_tokens = self.tof_embed(tof)
        mra_tokens = self.mra_embed(mra)
        
        orig_tokens = self.scale_orig_embed(scale_orig)
        down_tokens = self.scale_down_embed(scale_down)
        up_tokens = self.scale_up_embed(scale_up)
        
        # Modal-specific encoding
        for block in self.modal_encoder:
            tof_tokens = block(tof_tokens)
            mra_tokens = block(mra_tokens)
        
        # Scale-specific encoding
        for block in self.scale_encoder:
            orig_tokens = block(orig_tokens)
            down_tokens = block(down_tokens)
            up_tokens = block(up_tokens)
        
        # Cross-modal attention (bidirectional)
        tof_to_mra = self.cross_modal_attn(tof_tokens, mra_tokens)
        mra_to_tof = self.cross_modal_attn(mra_tokens, tof_tokens)
        
        # Cross-scale attention
        orig_to_down = self.cross_scale_attn(orig_tokens, down_tokens)
        orig_to_up = self.cross_scale_attn(orig_tokens, up_tokens)
        down_to_orig = self.cross_scale_attn(down_tokens, orig_tokens)
        up_to_orig = self.cross_scale_attn(up_tokens, orig_tokens)
        
        # Extract CLS tokens
        tof_cls = tof_to_mra[:, 0]
        mra_cls = mra_to_tof[:, 0]
        orig_cls = orig_to_down[:, 0]
        down_cls = down_to_orig[:, 0]
        up_cls = up_to_orig[:, 0]
        orig_up_cls = orig_to_up[:, 0]
        
        # Concatenate all features
        fused_features = torch.cat([
            tof_cls, mra_cls, 
            orig_cls, down_cls, up_cls, orig_up_cls
        ], dim=-1)
        
        # Fusion
        fused = self.fusion(fused_features)
        
        # Classification
        logits = self.head(fused)
        
        return logits


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = SAMM2D(
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_classes=2,
        embed_dim=256,
        depth=3,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1
    )
    
    # Dummy inputs
    batch_size = 2
    tof = torch.randn(batch_size, 1, 224, 224)
    mra = torch.randn(batch_size, 1, 224, 224)
    scale_orig = torch.randn(batch_size, 1, 224, 224)
    scale_down = torch.randn(batch_size, 1, 224, 224)
    scale_up = torch.randn(batch_size, 1, 224, 224)
    
    # Forward pass
    logits = model(tof, mra, scale_orig, scale_down, scale_up)
    
    print(f"Model output shape: {logits.shape}")
    print(f"Total parameters: {count_parameters(model):,}")
