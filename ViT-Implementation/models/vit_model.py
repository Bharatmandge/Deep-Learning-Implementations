import torch 
import torch.nn as nn 
from .patch_embedding import PatchEmbedding
from .transformer_block import TransformerBlock 

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed  = PatchEmbedding(config.img_size, config.patch_size, config.in_channels, config.embed_dim)
        
        seq_length = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, seq_length, config.embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads)
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # 1. Patchinh 
        x = self.patch_embed(x)
        
        # 2 ADD CLS 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add Positional EMbedding 
        x = x + self.pos_embed
        
        # 4. Pass through Transformer Blocks 
        for block in self.blocks:
            x = block(x)
            
        # 5. Classification Head
        x = self.norm(x)
        cls_output = x[:, 0]
        
        out = self.head(cls_output)
        return out 