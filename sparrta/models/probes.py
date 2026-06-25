import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from .util import get_2d_sincos_pos_embed, Attention


class ClassificationHead(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 num_classes: int,
                 use_layernorm: bool = True,
                 dropout_rate: float = 0.0,
                 attention_map = None,
                 head_type: str = "linear"):
        super().__init__()
        self.name = f"cls_{head_type}"
        self.attention_map = attention_map
        self.dropout_rate = dropout_rate
        self.use_layernorm = use_layernorm
        self.norm = nn.LayerNorm(feat_dim) if use_layernorm else None
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, feats):
        # feats is expected to be [B, D] (e.g., CLS token)
        if isinstance(feats, (list, tuple)):
            # If provided as a list of tensors, concatenate along feature dim
            feats = torch.cat(feats, dim=-1)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        if self.norm is not None:
            feats = self.norm(feats)
        feats = self.dropout(feats)
        return self.classifier(feats)

class EfficientProbing(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 8, # camera 4, human 4
        d_out: int = 16, # camera 8, human 8
        use_layernorm: bool = False,
        dropout_rate: float = 0.0,
        attention_map = None,
        head_type: str = "efficient",
    ):
        super().__init__()
        self.name = f'cls_{head_type}'
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.use_layernorm = use_layernorm
        self.feat_dim = feat_dim
        head_dim = feat_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.attention_map = attention_map
        self.d_out = d_out
        self.num_queries = num_queries
        self.norm = nn.LayerNorm(feat_dim // d_out) if use_layernorm else None
        self.v = nn.Linear(self.feat_dim, self.feat_dim // d_out, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, self.feat_dim) * 0.02)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.feat_dim // d_out, num_classes, bias=True)
        
    def forward(self, feats, cls=None):
        try:
            # from [B, (NxC)] to [B, N, C]
            feats = feats.view(feats.size(0),feats.size(1)//self.feat_dim,self.feat_dim)
        except:
            feats = feats
        cls_token = feats[:, 0]
        # feats = feats[:, 1:]
        B, N, C = feats.shape
        C_prime = C // self.d_out

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = (feats.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        q = q * self.scale
        v = (self.v(feats).reshape(B, N, self.num_queries, C // (self.d_out * self.num_queries)).permute(0, 2, 1, 3))

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attention_map = attn.squeeze(1)
        x_cls = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        x_cls = x_cls.view(B, C_prime)
        if self.norm is not None:
            x_cls = self.norm(x_cls)
        x_cls = self.proj_drop(x_cls)
        x_cls = self.classifier(x_cls)
        return x_cls

class ABMILPHead(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            num_classes: int,
            self_attention_apply_to: str = "none", # choices=["none", "map", "both"]
            activation: str= "relu",  # choices=["tanh", "relu"]
            depth: int = 1,
            cond: str="none", # choices=["none", "pe"]
            content: str = "all", # choices=["all", "patch"]
            num_patches: Optional[int] = None,
            use_layernorm: bool = False,
            dropout_rate: float = 0.0,
            attention_map = None,
            head_type: str = "abmilp",

        ):
        super().__init__()
        self.name = f'cls_{head_type}'
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.feat_dim = feat_dim
        self.cond = cond
        self.self_attention_apply_to = self_attention_apply_to
        self.content = content
        # num_patches = 256 if num_patches is None else num_patches # 256 for Large, 196 for Base models
        if self.cond == "pe":
            self.pos_embed = torch.nn.Parameter(
                torch.from_numpy(
                    get_2d_sincos_pos_embed(feat_dim, int(num_patches ** .5), cls_token=(content != "patch"))
                ).float().unsqueeze(0),
                requires_grad=False
            )
        else:
            self.pos_embed = None

        self.self_attn = Attention(feat_dim, num_heads=1) if self.self_attention_apply_to != "none" else nn.Identity()


        self.num_queries = 1

        attn_pred_layers = []
        for i in range(depth-1):
            attn_pred_layers.extend([
                nn.Linear(feat_dim, feat_dim),
                (nn.Tanh() if activation == "tanh" else nn.ReLU()),
            ])

        attn_pred_layers.append(nn.Linear(feat_dim, self.num_queries))
        self.attention_predictor = nn.Sequential(*attn_pred_layers)
        self.norm = nn.LayerNorm(self.feat_dim) if use_layernorm else None
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.feat_dim, num_classes)
        self.attention_map = attention_map
    def forward_with_attn_map(self, x):
        try:
            # from [B, (NxC)] to [B, N, C]
            x = x.view(x.size(0),x.size(1)//self.feat_dim,self.feat_dim)
        except:
            x = x

        if self.content == "patch":
            x = x[:, 1:] # keep patch tokens only

        x_attn = self.self_attn(x)
        if isinstance(x_attn, tuple):
            x_attn = x_attn[0]

        predictor_input = x_attn if self.self_attention_apply_to in ["map", "both"] else x

        if self.cond == "pe":
            predictor_input = predictor_input + self.pos_embed

        attn_map = self.attention_predictor(predictor_input)
        attn_map = F.softmax(attn_map, dim=1)
        self.attention_map = attn_map.permute(0, 2, 1)
        attn_map = self.attn_drop(attn_map)
        x_out = x_attn if self.self_attention_apply_to in ["both"] else x
        out = (x_out * attn_map).sum(dim=1)
        return out, attn_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cls, _ = self.forward_with_attn_map(x)
        if self.norm is not None:
            x_cls = self.norm(x_cls)
        x_cls = self.proj_drop(x_cls)
        x_cls = self.classifier(x_cls)
        return x_cls