# generation/eval_text.py (Modified to use custom trained classifier)
#!/usr/bin/env python
"""
Evaluate generated text for multiple target emotions using a custom-trained
classifier, addressing the label mismatch issue.

Expects an input directory containing subdirectories like 'generated_{emotion_name}',
each containing a file like 'all_{emotion_name}.txt'.

Outputs a single JSON file containing evaluation metrics for each target emotion.

Usage:
    python generation/eval_text.py \
      --input_dir results/generation_output \
      --evaluator_model_path results/emotion_classifier_model/best_model \
      --output_metrics results/all_emotions_eval_custom.json \
      [--batch_size 16]
"""
import argparse
import json
import os
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from sklearn.metrics import precision_recall_fscore_support

# Suppress some Hugging Face warnings if desired
hf_logging.set_verbosity_error()

# Define the target labels CONSISTENTLY with preprocess.py and train_evaluator.py
TARGET_10_LABELS = [
    "neutral", "admiration", "gratitude", "curiosity", "love",
    "anger", "joy", "sadness", "amusement", "confusion"
]
EMOTION_TO_IDX = {name: i for i, name in enumerate(TARGET_10_LABELS)}
NUM_LABELS = len(TARGET_10_LABELS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_texts(path):
    """Loads non-empty lines from a text file."""
    try:
        with open(path, "r", encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"  Error: File not found at {path}")
        return None
    except Exception as e:
        print(f"  Error reading file {path}: {e}")
        return None

def classify_texts_custom(texts, model, tokenizer, batch_size=16):
    """
    Classifies texts using the custom trained multi-label model.
    Returns multi-hot encoded predictions. Includes enhanced debugging.
    """
    model.eval() # Ensure model is in eval mode
    all_preds = []
    problematic_batch_indices = [] # Keep track of batches causing errors

    # --- Debug: Check tokenizer max length ---
    try:
        max_len = tokenizer.model_max_length
        print(f"  [Debug] Tokenizer model_max_length: {max_len}")
        if max_len is None or max_len > 1024: # Set a reasonable upper bound if needed
             print(f"  [Debug] Warning: model_max_length is large or None. Using 512.")
             max_len = 512
    except Exception as e:
        print(f"  [Debug] Error getting model_max_length: {e}. Using 512.")
        max_len = 512
    # --- End Debug ---

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"\n--- Processing Batch {i // batch_size + 1} (Indices {i} to {i+batch_size-1}) ---") # Print batch info
        # --- Debug: Print raw batch text ---
        # print(f"  [Debug] Batch Text:\n{batch_texts}\n")
        # --- End Debug ---

        inputs = None # Initialize inputs
        try:
            # --- Tokenization Step ---
            print("  Tokenizing batch...")
            inputs = tokenizer(
                batch_texts,
                padding=True, # Pad to longest in batch (or max_len if needed)
                truncation=True,
                max_length=max_len, # Use determined max_len
                return_tensors="pt"
                # Add return_overflowing_tokens=True might help debug truncation issues, but complicates output
            )
            # --- Debug: Print input IDs (first few tokens) ---
            print(f"  [Debug] Tokenization successful. Input IDs shape: {inputs['input_ids'].shape}")
            # print(f"  [Debug] Sample Input IDs (first 10):\n{inputs['input_ids'][:, :10]}")
            # print(f"  [Debug] Max Input ID in batch: {torch.max(inputs['input_ids'])}") # Check for huge IDs
            # --- End Debug ---

            inputs = inputs.to(DEVICE) # Move to device AFTER successful tokenization

        except Exception as e_tok:
            print(f"  ERROR during TOKENIZATION in batch {i // batch_size + 1}: {e_tok}")
            print(f"  Problematic Text Batch (potentially):\n{batch_texts}\n")
            problematic_batch_indices.append(i // batch_size + 1)
            # Append zero predictions for this failed batch
            batch_preds = [[0] * NUM_LABELS] * len(batch_texts)
            all_preds.extend(batch_preds)
            continue # Skip model inference for this batch

        try:
            # --- Model Inference Step ---
            print("  Running model inference...")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            print("  Model inference successful.")

            # --- Post-processing Step ---
            print("  Post-processing predictions...")
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(logits)
            batch_preds = (probs >= 0.5).cpu().numpy().astype(int)
            print("  Post-processing successful.")
            all_preds.extend(batch_preds)

        except Exception as e_inf:
            # Catch the specific error if possible, otherwise general Exception
            error_type = type(e_inf).__name__
            print(f"  ERROR during MODEL INFERENCE or POST-PROCESSING in batch {i // batch_size + 1}: {error_type}: {e_inf}")
            # If the error is 'int too big to convert', it likely happened here or during tokenization
            print(f"  Inputs that might have caused the error (shape {inputs['input_ids'].shape}):\n{inputs['input_ids']}\n")
            problematic_batch_indices.append(i // batch_size + 1)
            # Append zero predictions for this failed batch
            batch_preds = [[0] * NUM_LABELS] * len(batch_texts)
            all_preds.extend(batch_preds)
            continue # Move to next batch

    if problematic_batch_indices:
        print(f"\n[Warning] Errors occurred during classification for batches: {sorted(list(set(problematic_batch_indices)))}")
        print("[Warning] Metrics for affected emotions might be inaccurate (showing 0s for failed batches).")

    # Ensure the number of predictions matches the number of input texts
    if len(all_preds) != len(texts):
         print(f"\n[Critical Error] Mismatch in prediction count! Expected {len(texts)}, got {len(all_preds)}. Returning empty predictions.")
         return [[0] * NUM_LABELS] * len(texts) # Return placeholder

    return all_preds

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated text using a custom multi-label emotion classifier.")
    parser.add_argument("--input_dir", required=True, type=Path,
                        help="Base directory containing generated text subdirectories (e.g., 'results/generation_output').")
    # --- MODIFIED --- : Path to the custom model
    parser.add_argument("--evaluator_model_path", required=True, type=Path,
                        help="Path to the fine-tuned evaluator model directory (e.g., 'results/emotion_classifier_model/best_model').")
    # --- MODIFIED --- : Output file name changed slightly
    parser.add_argument("--output_metrics", default="results/all_emotions_eval_custom.json", type=Path,
                        help="Path to save the consolidated JSON evaluation metrics using the custom evaluator.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: Input directory not found or is not a directory: {args.input_dir}")
        return
    if not args.evaluator_model_path.is_dir():
         print(f"Error: Evaluator model directory not found: {args.evaluator_model_path}")
         return

    print(f"Loading custom evaluator model from: {args.evaluator_model_path}...")
    try:
        # --- MODIFIED --- : Load custom model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.evaluator_model_path).to(DEVICE)
        print(f"Custom evaluator loaded successfully. Using device: {DEVICE}")
    except Exception as e:
        print(f"Error loading custom evaluator model: {e}")
        return

    all_results = {}
    print(f"\nProcessing emotions from directory: {args.input_dir}")

    for emotion_name in TARGET_10_LABELS:
        print(f"--- Processing Target Emotion: {emotion_name} ---")
        target_emotion_idx = EMOTION_TO_IDX[emotion_name]
        current_file_path = args.input_dir / f"generated_{emotion_name}" / f"all_{emotion_name}.txt"

        texts = load_texts(current_file_path)

        if texts is None:
            print(f"  Skipping {emotion_name} due to file loading error.")
            all_results[emotion_name] = {"error": "Failed to load generated text file."}
            continue
        if not texts:
             print(f"  Skipping {emotion_name} because the file is empty or contains no valid text.")
             all_results[emotion_name] = {"error": "Generated text file is empty."}
             continue

        print(f"  Loaded {len(texts)} texts for target {emotion_name}. Classifying...")

        # --- MODIFIED --- : Generate multi-hot y_true and y_pred
        # y_true: Assume the *only* true label for these generated texts is the target one
        # Shape: (num_samples, num_labels)
        y_true_multi_hot = np.zeros((len(texts), NUM_LABELS), dtype=int)
        y_true_multi_hot[:, target_emotion_idx] = 1

        # y_pred: Get multi-hot predictions from the custom classifier
        # Shape: (num_samples, num_labels)
        y_pred_multi_hot = classify_texts_custom(texts, model, tokenizer, args.batch_size)

        if len(y_pred_multi_hot) != len(texts):
             print(f"  Error: Number of predictions ({len(y_pred_multi_hot)}) does not match number of texts ({len(texts)}). Skipping metrics calculation.")
             all_results[emotion_name] = {"error": "Prediction count mismatch."}
             continue

        # Calculate metrics for the specific target emotion class
        # Use precision_recall_fscore_support without averaging to get per-class results
        p, r, f1, sup = precision_recall_fscore_support(
            y_true_multi_hot,
            y_pred_multi_hot,
            average=None, # Get per-class scores
            labels=list(range(NUM_LABELS)), # Consider all possible labels
            zero_division=0
        )

        # Extract metrics for the target emotion index
        metrics = {
            # p[target_emotion_idx]: Precision for the target class
            # r[target_emotion_idx]: Recall for the target class (How many of the target samples were correctly predicted with the target label?)
            # f1[target_emotion_idx]: F1 for the target class
            # sup[target_emotion_idx]: Support for the target class (Number of true instances for this class, which is len(texts))
            "precision_target": float(p[target_emotion_idx]),
            "recall_target":    float(r[target_emotion_idx]),
            "f1_target":        float(f1[target_emotion_idx]),
            "support_target":   int(sup[target_emotion_idx]), # Should be len(texts)
        }
        all_results[emotion_name] = metrics
        print(f"  Metrics for Target {emotion_name}: P={metrics['precision_target']:.4f}, R={metrics['recall_target']:.4f}, F1={metrics['f1_target']:.4f}, Support={metrics['support_target']}")

    print(f"\nSaving consolidated evaluation metrics to {args.output_metrics}...")
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    try:
        with open(args.output_metrics, "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print("✔ Successfully saved evaluation metrics.")
    except Exception as e:
         print(f"Error saving metrics file: {e}")

    # --- Print Summary Table to Console (Updated Labels) ---
    print("\n=== Evaluation Summary (Custom Evaluator) ===")
    max_emotion_len = max(len(e) for e in TARGET_10_LABELS) if TARGET_10_LABELS else 15
    header_format = f"{{:<{max_emotion_len}}} {{:<10}} {{:<10}} {{:<10}} {{:<10}}"
    row_format = f"{{:<{max_emotion_len}}} {{:<10.4f}} {{:<10.4f}} {{:<10.4f}} {{:<10d}}" # Format numbers

    print(header_format.format("Emotion", "Precision", "Recall", "F1-Score", "Support"))
    print("-" * (max_emotion_len + 43))

    for emotion, metrics in all_results.items():
        if "error" in metrics:
            error_msg = metrics['error'][:40] + '...' if len(metrics.get('error','')) > 40 else metrics.get('error','')
            print(f"{emotion:<{max_emotion_len}} Error: {error_msg}")
        else:
            # Use the metrics calculated for the specific target class
            print(row_format.format(
                emotion,
                metrics.get('precision_target', 0.0),
                metrics.get('recall_target', 0.0),
                metrics.get('f1_target', 0.0),
                metrics.get('support_target', 0)
            ))

    print("-" * (max_emotion_len + 43))
    # --- End Print Summary Table ---

if __name__ == "__main__":
    main()
