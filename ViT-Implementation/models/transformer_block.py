import torch 
import torch.nn as nn 
from .attention import AttentionMap

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = AttentionMap(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), 
            nn.GELU(), 
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x =x + mlp_out
        
        return x 
    
        