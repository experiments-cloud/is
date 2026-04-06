"""
Mechanistic Interpretability: Attention Dilution Heatmap
-------------------------------------------------------
Description:
    Extracts the internal self-attention matrices (Q*K^T) from an open-weight causal 
    LLM (e.g., Llama-3.1-8B) to visually demonstrate the "Attention Dilution" phenomenon.
    Generates the high-resolution heatmap utilized in the paper to show how probability 
    mass fragments across obsolete topological states within an append-only KV Cache.
    
Usage:
    Ensure the 'HF_TOKEN' environment variable is set before execution to authenticate
    and download gated models from Hugging Face.
"""

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Authenticate safely using environment variables
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("[!] Warning: HF_TOKEN not found in environment variables. Access to gated models may fail.")

# 1. Model Configuration and Loading
model_id = "meta-llama/Meta-Llama-3.1-8B"

print(f"Loading tokenizer and model: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# CRITICAL: output_attentions=True is required to extract the QK^T matrices
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16, 
    output_attentions=True 
)

# 2. Construct a test sequence (Simplified Stress Test)
# Simulates a history where R1 is updated multiple times to force contextual confusion
prompt = (
    "<scratchpad>\n"
    "Step 1: R1=0, R2=0\n"
    "Step 2: Add 5 to R1. R1=5\n"
    "Step 3: Add 2 to R1. R1=7\n"
    "Step 4: Subtract 1 from R1. R1=6\n"
    "Step 5: Copy R1 to R2. R2=" # The model should ideally attend to '6'
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 3. Forward Pass (No gradient calculation)
print("Executing inference (this may take a moment due to RAM offloading)...")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

# 4. Attention Matrix Extraction
attentions = outputs.attentions
layer_to_inspect = -1  # Final layer
head_to_inspect = 0    # First attention head
attention_matrix = attentions[layer_to_inspect][0, head_to_inspect, :, :].cpu().numpy()

# 5. "Flattening" Visualization (Publication-Ready)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
clean_tokens = [t.replace('Ġ', '').replace('<|begin_of_text|>', '<BOS>') for t in tokens]

# Canvas setup for horizontal stretching
plt.figure(figsize=(22, 10), dpi=300)

ax = sns.heatmap(
    attention_matrix, 
    xticklabels=clean_tokens, 
    yticklabels=clean_tokens, 
    cmap="inferno",
    cbar_kws={'label': 'Softmax Probability'}
)

plt.title(f"Mechanistic Extraction: Attention Dilution (Layer {layer_to_inspect}, Head {head_to_inspect})\nProbability mass fragments across obsolete states prior to state resolution", fontsize=16, pad=25)
plt.xlabel("Key (Past Context mapped in KV Cache)", fontsize=14, labelpad=15)
plt.ylabel("Query (Current Auto-regressive Token)", fontsize=14, labelpad=15)

# Aggressive font size control and rotation for readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

# Manual margin adjustments to prevent label cropping
plt.subplots_adjust(bottom=0.25, left=0.15)
plt.tight_layout()

output_filename = "attention_heatmap_ready.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Image rendered and saved successfully as '{output_filename}'")