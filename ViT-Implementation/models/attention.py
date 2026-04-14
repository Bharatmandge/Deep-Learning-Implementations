import torch 
import torch.nn as nn 

class AttentionMap(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        attn_output, attn_weights = self.mha(query=x, key=x, value=x)
        return attn_output, attn_weights
      