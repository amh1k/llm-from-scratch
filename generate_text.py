import torch
import torch.nn as nn




def generate_text(model, idx, max_tokens, context_size):
    for  _ in range(max_tokens):
        idx_cond = idx[:, -context_size:]
        # print("idx_cond shape: ", idx_cond.shape)
        with torch.no_grad():
            logits = model(idx_cond)
        # print("Shape of logits before slicing: ", logits.shape)
            
        logits =logits[:,-1,:] #Shape of logits is (batches, no of tokens, vocab_size)
        #We only want last token so we do -1
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # print(idx.shape)
        # print(idx_next.shape)
        
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx


def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special= {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_to_text(token_ids, tokenizer):
    flat = token_ids.flatten()
    return tokenizer.decode(flat.tolist())