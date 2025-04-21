# generation/gpt2_inject.py (Modified)
#!/usr/bin/env python
"""
Generate text with GPT-2 by injecting a single emotion vector
(selected from a probe matrix) into a specified hidden layer.

Usage:
    # Example: Inject 'anger' (index 5) vector from layer 10
    python generation/gpt2_inject.py \
      --prompt "Tell me a story" \
      --vector_matrix probe_model/vectors/layer_10.pt \
      --emotion_name anger \
      --layer 10 \
      --output_file out_anger.txt
"""
import argparse
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path

# Define the target labels and their order consistently
# (Must match the order used during probe training/extraction)
TARGET_10_LABELS = [
    "neutral", "admiration", "gratitude", "curiosity", "love",
    "anger", "joy", "sadness", "amusement", "confusion"
]
EMOTION_TO_IDX = {name: i for i, name in enumerate(TARGET_10_LABELS)}

# Global variable to hold the vector for the hook
# (Simpler than functools.partial for this case)
_GLOBAL_EMOTION_VEC = None

def hook_fn(module, inp, out):
    """Adds the _GLOBAL_EMOTION_VEC to the hidden states."""
    global _GLOBAL_EMOTION_VEC
    if _GLOBAL_EMOTION_VEC is None:
        return out

    # Output of transformer block is usually (hidden_states, present_key_value, ...)
    # Or just hidden_states for older/different configs
    if isinstance(out, tuple):
        hidden_states = out[0]
        rest = out[1:]
        modified_hidden_states = hidden_states + _GLOBAL_EMOTION_VEC.to(hidden_states.device)
        return (modified_hidden_states,) + rest
    else:
        # Assuming 'out' is the hidden_states tensor directly
        hidden_states = out
        modified_hidden_states = hidden_states + _GLOBAL_EMOTION_VEC.to(hidden_states.device)
        return modified_hidden_states


def generate_with_emotion(prompt, matrix_path, emotion_name, layer_idx,
                          output_file, max_length, temperature,
                          model_name):
    global _GLOBAL_EMOTION_VEC
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Vector Loading and Selection ---
    if emotion_name not in EMOTION_TO_IDX:
        raise ValueError(f"Unknown emotion '{emotion_name}'. Choose from: {TARGET_10_LABELS}")
    emotion_idx = EMOTION_TO_IDX[emotion_name]

    if not Path(matrix_path).exists():
        raise FileNotFoundError(f"Vector matrix file not found: {matrix_path}")

    # Load the entire matrix W (10, hidden_dim_probe)
    W_matrix = torch.load(matrix_path, map_location="cpu") # Load to CPU first
    if W_matrix.shape[0] != len(TARGET_10_LABELS):
         raise ValueError(f"Matrix {matrix_path} has {W_matrix.shape[0]} rows, expected {len(TARGET_10_LABELS)}.")

    # Select the specific emotion vector (row)
    emot_vec_probe = W_matrix[emotion_idx] # Shape: (hidden_dim_probe,) e.g., (1024,)
    # --- End Vector Loading ---

    # --- Load GPT-2 ---
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Add padding token if missing (GPT-2 usually doesn't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    # --- End Load GPT-2 ---

    # --- Projection ---
    hidden_dim_probe = emot_vec_probe.shape[-1]
    hidden_dim_gpt = model.config.n_embd

    if hidden_dim_probe != hidden_dim_gpt:
        print(f"Projecting vector from {hidden_dim_probe} to {hidden_dim_gpt} dimensions.")
        proj_layer = nn.Linear(hidden_dim_probe, hidden_dim_gpt, bias=False).to(device)
        # Initialize projection layer (optional, but can help)
        # nn.init.xavier_uniform_(proj_layer.weight)
        emot_vec_gpt = proj_layer(emot_vec_probe.to(device)) * 2 # Project and move to device
    else:
        emot_vec_gpt = emot_vec_probe.to(device) # Just move to device
    # --- End Projection ---

    # --- Injection Hook ---
    _GLOBAL_EMOTION_VEC = emot_vec_gpt # Set global for hook
    try:
        # Ensure layer index is valid
        if not (0 <= layer_idx < len(model.transformer.h)):
             raise ValueError(f"Invalid layer index {layer_idx}. Model has {len(model.transformer.h)} layers (0 to {len(model.transformer.h)-1}).")

        # Register the hook on the specified transformer block's output
        hook_handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
        print(f"Registered hook on layer {layer_idx}")
    except Exception as e:
        _GLOBAL_EMOTION_VEC = None # Clean up global vec on error
        raise RuntimeError(f"Failed to register hook: {e}")
    # --- End Injection Hook ---

    # --- Generation ---
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length or 512).to(device)

    print(f"Generating text (max_length={max_length}, temperature={temperature})...")
    # Generate text
    with torch.no_grad(): # Ensure no gradients are computed
        outs = model.generate(
            **inputs,
            max_new_tokens=max_length - inputs['input_ids'].shape[1], # Generate up to max_length TOTAL tokens
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id # Important for batch generation/padding
        )
    txt = tokenizer.decode(outs[0], skip_special_tokens=True)
    # --- End Generation ---

    # --- Cleanup ---
    hook_handle.remove()
    _GLOBAL_EMOTION_VEC = None # Reset global variable
    print("Removed hook.")
    # --- End Cleanup ---

    # --- Save Output ---
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(txt)
    print(f"✔ Generated text saved to {output_path}")
    # --- End Save Output ---


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2 using a single injected emotion vector.")
    parser.add_argument("--prompt", required=True, help="Input prompt for generation.")
    parser.add_argument("--vector_matrix", required=True,
                        help="Path to the .pt file containing the (10, hidden_dim) probe weight matrix.")
    parser.add_argument("--emotion_name", required=True, choices=TARGET_10_LABELS,
                        help="Name of the emotion vector to select from the matrix.")
    parser.add_argument("--layer", type=int, required=True, help="Index of the GPT-2 transformer layer to inject into (e.g., 0-11 for gpt2, 0-23 for gpt2-medium etc.).")
    parser.add_argument("--output_file", default="generated.txt", help="Path to save the generated text.")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum total length (prompt + generated) of the output sequence.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation (higher = more random).")
    parser.add_argument("--model_name", default="gpt2-medium", help="HuggingFace model name or path (e.g., gpt2, gpt2-medium).")
    args = parser.parse_args()

    generate_with_emotion(
        prompt=args.prompt,
        matrix_path=args.vector_matrix,
        emotion_name=args.emotion_name,
        layer_idx=args.layer,
        output_file=args.output_file,
        max_length=args.max_length,
        temperature=args.temperature,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()
