# import torch
# import torch.nn as nn
# from transformers import GPT2Model, GPT2Config, GPT2Tokenizer


# # Define the tiled matrix multiplication function
# def tiled_matmul(A, B, tile_size):
#     batch_size, num_heads, n, m = A.shape
#     m, p = B.shape[-2], B.shape[-1]
    
#     # Initialize the output matrix
#     C = torch.zeros((batch_size, num_heads, n, p), device=A.device)
    
#     # Iterate over tiles
#     for i in range(0, n, tile_size):
#         for j in range(0, p, tile_size):
#             for k in range(0, m, tile_size):
#                 # Define the tile sub-matrices
#                 A_tile = A[:, :, i:i+tile_size, k:k+tile_size]
#                 B_tile = B[:, :, k:k+tile_size, j:j+tile_size]
                
#                 # Perform multiplication on tiles
#                 C[:, :, i:i+tile_size, j:j+tile_size] += torch.matmul(A_tile, B_tile)
                
#     return C

# # Define the custom TiledAttention layer
# class TiledAttention(nn.Module):
#     def __init__(self, hidden_size, num_attention_heads, tile_size):
#         super(TiledAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_attention_heads = num_attention_heads
#         self.tile_size = tile_size

#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size, hidden_size)
#         self.value = nn.Linear(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, hidden_size)
        
#         self.softmax = nn.Softmax(dim=-1)
#         self.scale = 1.0 / (hidden_size // num_attention_heads) ** 0.5

#     def forward(self, x):
#         batch_size, seq_length, hidden_size = x.size()

#         # Linear projections
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
        
#         # Reshape for multi-head attention
#         q = q.view(batch_size, seq_length, self.num_attention_heads, hidden_size // self.num_attention_heads).transpose(1, 2)
#         k = k.view(batch_size, seq_length, self.num_attention_heads, hidden_size // self.num_attention_heads).transpose(1, 2)
#         v = v.view(batch_size, seq_length, self.num_attention_heads, hidden_size // self.num_attention_heads).transpose(1, 2)
        
#         # Scaled dot-product attention
#         scores = tiled_matmul(q, k.transpose(-1, -2), self.tile_size) * self.scale
#         attention_weights = self.softmax(scores)
#         attention_output = tiled_matmul(attention_weights, v, self.tile_size)
        
#         # Concatenate heads
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        
#         # Final linear layer
#         output = self.out(attention_output)
        
#         return output

# # Extend the GPT-2 model to replace the attention layers with TiledAttention
# class GPT2WithTiledAttention(GPT2Model):
#     def __init__(self, config, tile_size):
#         super(GPT2WithTiledAttention, self).__init__(config)
#         self.tile_size = tile_size

#         # Replace the attention layers with TiledAttention
#         for block in self.h:
#             block.attn.c_attn = TiledAttention(config.n_embd, config.n_head, tile_size)

# # Initialize the GPT-2 configuration and model with tiled attention
# config = GPT2Config()
# tile_size = 64
# model = GPT2WithTiledAttention(config, tile_size)

# # Initialize the tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Sample sentence
# sentence = "Hello, this is a sample sentence to test the attention mechanism."
# input_ids = tokenizer.encode(sentence, return_tensors='pt')

# # Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# input_ids = input_ids.to(device)

# # Run the model
# with torch.no_grad():
#     output = model(input_ids)

# print(f"Output shape: {output.last_hidden_state.shape}")

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer


# Define the tiled matrix multiplication function
def tiled_matmul(A, B, tile_size):
    batch_size, num_heads, n, m = A.shape
    m, p = B.shape[-2], B.shape[-1]
    
    # Initialize the output matrix
    C = torch.zeros((batch_size, num_heads, n, p), device=A.device)
    
    # Iterate over tiles
    for i in range(0, n, tile_size):
        for j in range(0, p, tile_size):
            for k in range(0, m, tile_size):
                # Define the tile sub-matrices
                A_tile = A[:, :, i:i+tile_size, k:k+tile_size]
                B_tile = B[:, :, k:k+tile_size, j:j+tile_size]
                
                # Perform multiplication on tiles
                C[:, :, i:i+tile_size, j:j+tile_size] += torch.matmul(A_tile, B_tile)
                
    return C

# Define the custom TiledAttention layer
class TiledAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, tile_size):
        super(TiledAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.tile_size = tile_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.softmax = nn.Softmax(dim=-1)
        self.scale = 1.0 / (hidden_size // num_attention_heads) ** 0.5

    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Linear projections
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_attention_heads, hidden_size // self.num_attention_heads).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_attention_heads, hidden_size // self.num_attention_heads).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_attention_heads, hidden_size // self.num_attention_heads).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = tiled_matmul(q, k.transpose(-1, -2), self.tile_size) * self.scale
        attention_weights = self.softmax(scores)
        attention_output = tiled_matmul(attention_weights, v, self.tile_size)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        
        # Final linear layer
        output = self.out(attention_output)
        
        return output

# Extend the GPT-2 model to replace the attention layers with TiledAttention
class GPT2WithTiledAttention(GPT2Model):
    def __init__(self, config, tile_size):
        super(GPT2WithTiledAttention, self).__init__(config)
        self.tile_size = tile_size

        # Replace the attention layers with TiledAttention
        for block in self.h:
            block.attn.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
            block.attn.c_proj = nn.Linear(config.n_embd, config.n_embd)
            block.attn.split_size = config.n_embd

            # Replace the attention mechanism with TiledAttention
            block.attn.attn = TiledAttention(config.n_embd, config.n_head, tile_size)

# Initialize the GPT-2 configuration and model with tiled attention
config = GPT2Config()
tile_size = 64
model = GPT2WithTiledAttention(config, tile_size)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Sample sentence
sentence = "Hello, how are you?"
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)

# Run the model
with torch.no_grad():
    output = model(input_ids)

print(f"Output shape: {output.last_hidden_state.shape}")
