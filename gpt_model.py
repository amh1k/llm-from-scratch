import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers= nn.Sequential(nn.Linear(cfg['emb_dim'], cfg['emb_dim'] * 4), nn.GELU(),
        nn.Linear(cfg['emb_dim'] * 4,cfg['emb_dim']))
        #What we are doing is first projecting the embedding dimesion into a dimension which is 4*embd_dim then applying gelu
        #and then condesing the dimension back to the original embedding one
        
    def forward(self, X):
        return self.layers(X)
        
class LayerNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale= nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, X):
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (X - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
        


class TransformerBlock(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.transformer= MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out= cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'],
            hasBias=cfg['qkv_bias']
            
            
            
            
        )
        self.feed_forward= FeedForwardNetwork(cfg)
        self.norm1= LayerNormalization(cfg['emb_dim'])
        self.norm2= LayerNormalization(cfg['emb_dim'])
        self.drop_shortcut= nn.Dropout(cfg['drop_rate'])
        
        
    def forward(self, X):
        shortcut = X
        X = self.norm1(X)
        X = self.transformer(X)
        X=self.drop_shortcut(X)
        X=X+shortcut
        shortcut=X
        X=self.norm2(X)
        X = self.feed_forward(X)
        X=self.drop_shortcut(X)
        X=X+shortcut
        return X
        

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embeddings= nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.position_embeddings=nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_embeddings= nn.Dropout(cfg['drop_rate'])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm = LayerNormalization(cfg['emb_dim'])
        self.output_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len= in_idx.shape
        tok_embds= self.token_embeddings(in_idx)
        pos_embds = self.position_embeddings(torch.arange(seq_len, device=in_idx.device))
        x = tok_embds + pos_embds#adding the word embeddings and the positional embeddings
        x = self.drop_embeddings(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits
        
        