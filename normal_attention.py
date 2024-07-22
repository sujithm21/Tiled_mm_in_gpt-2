import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2', output_attentions=True)
model = GPT2Model.from_pretrained('gpt2', config=config)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sample sentence
sentence = "Hello, how are you?"
input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)

# Define a function to profile the attention layers
def profile_attention(inputs):
    def attention_hook(module, input, output):
        return output
    
    handles = []
    # Register hooks on attention layers
    for layer in model.h:
        handle = layer.attn.register_forward_hook(attention_hook)
        handles.append(handle)
    
    # Perform forward pass with profiling
    with torch.autograd.profiler.profile() as prof:
        outputs = model(inputs)
    
    # Remove hooks
    for handle in handles:
        handle.remove()

    # Save profiling results
    prof.export_chrome_trace("profiler_normal.json")

    # Print the profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Profile the attention layers
profile_attention(input_ids)

# Run the model
with torch.no_grad():
    output = model(input_ids)

print(f"Output shape (normal): {output.last_hidden_state.shape}")
