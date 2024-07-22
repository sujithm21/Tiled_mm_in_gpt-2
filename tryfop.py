import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer

# Optimized tiled feed-forward function
def tiled_feed_forward(X, W1, b1, W2, b2, tile_size):
    batch_size, seq_length, hidden_size = X.shape
    
    # Initialize the intermediate and output matrices
    intermediate = torch.zeros((batch_size, seq_length, W1.shape[1]), device=X.device)
    output = torch.zeros((batch_size, seq_length, W2.shape[1]), device=X.device)
    
    # Iterate over tiles efficiently for the first linear layer
    for i in range(0, seq_length, tile_size):
        for j in range(0, W1.shape[1], tile_size):
            for k in range(0, hidden_size, tile_size):
                X_tile = X[:, i:i+tile_size, k:k+tile_size]
                W1_tile = W1[k:k+tile_size, j:j+tile_size]
                intermediate[:, i:i+tile_size, j:j+tile_size] += torch.matmul(X_tile, W1_tile)
                
    intermediate += b1
    intermediate = torch.relu(intermediate)
    
    # Iterate over tiles efficiently for the second linear layer
    for i in range(0, seq_length, tile_size):
        for j in range(0, W2.shape[1], tile_size):
            for k in range(0, W1.shape[1], tile_size):
                intermediate_tile = intermediate[:, i:i+tile_size, k:k+tile_size]
                W2_tile = W2[k:k+tile_size, j:j+tile_size]
                output[:, i:i+tile_size, j:j+tile_size] += torch.matmul(intermediate_tile, W2_tile)
                
    output += b2
    
    return output

# Define the custom TiledFeedForward layer
class TiledFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, tile_size):
        super(TiledFeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tile_size = tile_size

        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Linear projections
        W1 = self.dense_1.weight.t()
        b1 = self.dense_1.bias
        W2 = self.dense_2.weight.t()
        b2 = self.dense_2.bias

        # Perform tiled feed-forward operation
        output = tiled_feed_forward(hidden_states, W1, b1, W2, b2, self.tile_size)

        return output

# Extend the GPT-2 model to replace the feed-forward layers with TiledFeedForward
class GPT2WithTiledFeedForward(GPT2Model):
    def __init__(self, config, tile_size):
        super().__init__(config)
        self.tile_size = tile_size

        # Replace the feed-forward layers with TiledFeedForward
        for block in self.h:
            hidden_size = config.n_embd
            intermediate_size = config.n_inner

            # Initialize the custom TiledFeedForward layer
            block.mlp.c_fc = nn.Identity()
            block.mlp.c_proj = nn.Identity()
            block.mlp.fc1 = TiledFeedForward(hidden_size, intermediate_size, tile_size)
            block.mlp.fc2 = nn.Identity()  # Identity layer since we handle the second linear layer in TiledFeedForward

# Initialize the GPT-2 configuration and model with tiled feed-forward layers
config = GPT2Config()
tile_size = 64
model = GPT2WithTiledFeedForward(config, tile_size)

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Sample sentence
sentence = "Hello, how are you?"
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Run the model
with torch.no_grad():
    output = model(input_ids)

print(f"Output shape: {output.last_hidden_state.shape}")
