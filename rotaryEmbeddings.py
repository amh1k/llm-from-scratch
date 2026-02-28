import torch
import torch.nn as nn

class RotaryEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__();
        self.inv_freq= 1.0/ (base ** (torch.arange(0, dim,2).float() / dim))
        self.register_buffer('inv_freq', self.inv_freq)
        self.max_seq_len=max_seq_len
    def forward(self, x, seq_len):
        # x shape: (batch, num_heads, seq_len, head_dim)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs=torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, dim) since inv_freq was half of dim
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    
def apply_rotary_emb(x, cos, sin):
    def rotate_half(x):
        x1 = x[..., :x.shape[-1]//2]
        x2= x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    return(x * cos) + (rotate_half(x) * sin)
    