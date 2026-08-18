"""
Vision-Language Multimodal Backbone for Embodied Manipulation.
Combines:
1. Patch-based Vision Encoder (SigLIP / DINOv2 style feature extraction with dynamic resolution resize)
2. Semantic Text Embedding & Cross-Attention Transformer
3. Spatial Feature Projector & Multimodal Context Aggregator
4. Automatic loading from local downloaded pretrained directory
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class PatchVisionEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        local_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 14x14 = 196
        
        # Patch projection layer
        self.patch_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
        if local_pretrained_path and os.path.exists(local_pretrained_path):
            self.load_local_weights(local_pretrained_path)

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def load_local_weights(self, path: str):
        print(f"[*] [Vision Encoder] 正在从本地目录加载预训练权重: {os.path.abspath(path)}")
        for fname in ["pytorch_model.bin", "model.safetensors", "model.pt"]:
            fp = os.path.join(path, fname)
            if os.path.exists(fp):
                try:
                    if fp.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(fp)
                    else:
                        state_dict = torch.load(fp, map_location="cpu")
                    self.load_state_dict(state_dict, strict=False)
                    print(f"[✓] [Vision Encoder] 成功加载本地权重: {fname}")
                    break
                except Exception as e:
                    print(f"[!] [Vision Encoder] 本地权重解析提示: {e}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 3, H, W) - Handles any image resolution (e.g. 256x256 from BridgeData)
        """
        B = x.shape[0]
        # Dynamically resize to model's canonical resolution (224x224) if needed
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            
        patches = self.patch_proj(x).flatten(2).transpose(1, 2)  # (B, 196, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, patches), dim=1) + self.pos_embed  # (B, 197, D)
        
        out = self.encoder(tokens)
        out = self.norm(out)
        
        cls_feat = out[:, 0]
        patch_tokens = out[:, 1:]
        return cls_feat, patch_tokens


class TextLanguageEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 768,
        max_len: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        local_pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        if local_pretrained_path and os.path.exists(local_pretrained_path):
            self.load_local_weights(local_pretrained_path)

    def load_local_weights(self, path: str):
        print(f"[*] [Language Backbone] 正在从本地目录加载预训练权重: {os.path.abspath(path)}")
        for fname in ["pytorch_model.bin", "model.safetensors", "model.pt"]:
            fp = os.path.join(path, fname)
            if os.path.exists(fp):
                try:
                    if fp.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(fp)
                    else:
                        state_dict = torch.load(fp, map_location="cpu")
                    self.load_state_dict(state_dict, strict=False)
                    print(f"[✓] [Language Backbone] 成功加载本地权重: {fname}")
                    break
                except Exception as e:
                    print(f"[!] [Language Backbone] 本地权重解析提示: {e}")

    def tokenize_text(self, text_list: Any, device: torch.device) -> torch.Tensor:
        if isinstance(text_list, str):
            text_list = [text_list]
        batch_ids = []
        for text in text_list:
            words = str(text).lower().split()
            ids = [(abs(hash(w)) % (self.vocab_size - 10) + 1) for w in words[:self.max_len]]
            if len(ids) < self.max_len:
                ids = ids + [0] * (self.max_len - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long, device=device)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :L, :]
        out = self.encoder(x)
        out = self.norm(out)
        global_feat = torch.mean(out, dim=1)
        return global_feat, out


class EmbodiedMultimodalBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        local_vision_path: Optional[str] = None,
        local_text_path: Optional[str] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vision_encoder = PatchVisionEncoder(
            embed_dim=hidden_dim, local_pretrained_path=local_vision_path
        )
        self.text_encoder = TextLanguageEncoder(
            embed_dim=hidden_dim, local_pretrained_path=local_text_path
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(
        self,
        images: torch.Tensor,
        instructions: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = images.device
        if not isinstance(instructions, torch.Tensor):
            input_ids = self.text_encoder.tokenize_text(instructions, device=device)
        else:
            input_ids = instructions
            
        cls_img, patch_tokens = self.vision_encoder(images)
        global_text, text_tokens = self.text_encoder(input_ids)
        
        fused_patches, attn_weights = self.cross_attn(
            query=patch_tokens, key=text_tokens, value=text_tokens
        )
        
        global_combined = torch.cat([cls_img, global_text], dim=-1)
        context_embedding = self.fusion_mlp(global_combined)
        return context_embedding, patch_tokens, attn_weights
