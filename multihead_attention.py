import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,dropout, num_heads, hasBias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        self.d_out=d_out
        self.num_heads = num_heads
        self.head_dim = d_out//self.num_heads
        self.W_key=nn.Linear(d_in, d_out, bias=hasBias)
        self.W_value=nn.Linear(d_in, d_out, bias=hasBias)
        self.W_query=nn.Linear(d_in, d_out, bias=hasBias)
        #We use a linear lyaer to combine outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",torch.triu(torch.ones(context_length, context_length),diagonal=1))
        
    def forward(self, X):
        b,no_tokens,d_in = X.shape
        keys= self.W_key(X)
        values=self.W_value(X)
        queries=self.W_query(X)
        keys= keys.view(b, no_tokens, self.num_heads, self.head_dim)
        values= values.view(b, no_tokens, self.num_heads, self.head_dim)
        queries= queries.view( b, no_tokens, self.num_heads, self.head_dim)
       
        keys = keys.transpose(1,2) #From (no of batches, no of tokens, no of heads, head_dim) to (no of batches, no of heads, no of tokens, head_dim)
        values = values.transpose(1,2)
        
        queries=queries.transpose(1,2)
        temp=keys.transpose(2,3)
        attn_scores= queries @ temp
        
        #(batches, no_heads, no_tokens, head_dim)*(batches, no_heads, head_dim, no_tokens)
        #Matrix multiplication occurs between the last 2 dimensions
        mask_bool = self.mask.bool()[:no_tokens, :no_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)#This gives one Attention Map (how every word relates to every other word) per head, per batch.
        attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim=-1)
        attn_weights=self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b, no_tokens, self.d_out)
        
        #We add optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec

        
        
        