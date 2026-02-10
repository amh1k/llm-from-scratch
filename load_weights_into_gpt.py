import numpy as np
import torch
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
        "Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    # Load base embeddings
    gpt.position_embeddings.weight = assign(gpt.position_embeddings.weight, params['wpe'])
    gpt.token_embeddings.weight = assign(gpt.token_embeddings.weight, params['wte'])
    
    # Iterate through each transformer block
    for b in range(len(params["blocks"])):
        # Split the combined attention weights 
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.transformer_blocks[b].transformer.W_query.weight = assign(
            gpt.transformer_blocks[b].transformer.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].transformer.W_key.weight = assign(
            gpt.transformer_blocks[b].transformer.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].transformer.W_value.weight = assign(
            gpt.transformer_blocks[b].transformer.W_value.weight, v_w.T)
        
        # Split the combined attention biases
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.transformer_blocks[b].transformer.W_query.bias = assign(
            gpt.transformer_blocks[b].transformer.W_query.bias, q_b)
        gpt.transformer_blocks[b].transformer.W_key.bias = assign(
            gpt.transformer_blocks[b].transformer.W_key.bias, k_b)
        gpt.transformer_blocks[b].transformer.W_value.bias = assign(
            gpt.transformer_blocks[b].transformer.W_value.bias, v_b)
        
        # Output projection in transformerention
        gpt.transformer_blocks[b].transformer.out_proj.weight = assign(
            gpt.transformer_blocks[b].transformer.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].transformer.out_proj.bias = assign(
            gpt.transformer_blocks[b].transformer.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        # Feed-Forward Network (MLP) layers
        gpt.transformer_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        # Layer Normalization 1 (Pre-Attention)
        gpt.transformer_blocks[b].norm1.scale = assign(
            gpt.transformer_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].norm1.shift = assign(
            gpt.transformer_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        
        # Layer Normalization 2 (Pre-FFN)
        gpt.transformer_blocks[b].norm2.scale = assign(
            gpt.transformer_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].norm2.shift = assign(
            gpt.transformer_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
            
    # Final Layer Normalization and Output Head
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.output_head.weight = assign(gpt.output_head.weight, params["wte"])