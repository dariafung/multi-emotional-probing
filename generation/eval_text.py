# generation/eval_text.py (Modified to handle multiple emotions via directory)
#!/usr/bin/env python
"""
Evaluate generated text for multiple target emotions based on a directory structure.
Expects an input directory containing subdirectories named like 'generated_{emotion_name}',
each containing a file like 'all_{emotion_name}.txt'.

Outputs a single JSON file containing evaluation metrics for each successfully evaluated emotion.

Usage:
    python generation/eval_text.py \
      --input_dir results/generation_output \
      --output_metrics results/all_emotions_eval.json \
      [--model_name bhadresh-savani/bert-base-uncased-emotion]
"""
import argparse
import json
import os
from pathlib import Path
from transformers import pipeline, logging as hf_logging
from sklearn.metrics import precision_recall_fscore_support

# Suppress some Hugging Face warnings if desired
hf_logging.set_verbosity_error()

# Define the target labels consistently
TARGET_10_LABELS = [
    "neutral", "admiration", "gratitude", "curiosity", "love",
    "anger", "joy", "sadness", "amusement", "confusion"
]

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

def classify_texts(texts, classifier, batch_size=16):
    """Classifies texts using the pipeline, handling potential errors."""
    preds = []
    try:
        # Process in batches for potentially better performance
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = classifier(batch) # Classifier might return list of lists or list of dicts
            for result in results:
                 # Handle different pipeline output formats
                 if isinstance(result, list): # Newer pipelines might return list of score dicts per item
                     scores = result
                 else: # Older might return single score dict
                     scores = [result] # Should be checked if this case is correct

                 # Find the label with the highest score
                 if scores: # Check if scores list is not empty
                    pred_label = max(scores, key=lambda x: x["score"])["label"].lower()
                    preds.append(pred_label)
                 else:
                     preds.append("unknown") # Handle cases where classifier returns nothing

        # Original per-text classification (simpler but potentially slower)
        # for t in texts:
        #     # Wrap in try-except for resilience against single text errors
        #     try:
        #         scores = classifier(t)[0]  # list of {label, score}
        #         pred = max(scores, key=lambda x: x["score"])["label"].lower()
        #         preds.append(pred)
        #     except Exception as e:
        #         print(f"    Warning: Classifier error on text: '{t[:50]}...': {e}")
        #         preds.append("classifier_error") # Append a placeholder on error

    except Exception as e:
        print(f"  Error during batch classification: {e}")
        # Return empty list or partial list depending on where error occurred
        return ["batch_classification_error"] * len(texts)
    return preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated text for multiple emotions based on directory structure.")
    parser.add_argument("--input_dir", required=True, type=Path,
                        help="Base directory containing generated text subdirectories (e.g., 'results/generation_output').")
    parser.add_argument("--model_name", default="bhadresh-savani/bert-base-uncased-emotion",
                        help="HF model for emotion classification.")
    parser.add_argument("--output_metrics", default="results/all_emotions_eval.json", type=Path,
                        help="Path to save the consolidated JSON evaluation metrics.")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: Input directory not found or is not a directory: {args.input_dir}")
        return

    print(f"Loading classifier model: {args.model_name}...")
    try:
        # Ensure return_all_scores is set appropriately for the model if needed
        # Some models might inherently return all scores, others might need the flag
        classifier = pipeline(
            "text-classification",
            model=args.model_name,
            return_all_scores=True # Keep this, usually safe
        )
        print("Classifier loaded.")
    except Exception as e:
        print(f"Error loading classifier model: {e}")
        return

    all_results = {}
    print(f"\nProcessing emotions from directory: {args.input_dir}")

    for emotion_name in TARGET_10_LABELS:
        print(f"--- Processing: {emotion_name} ---")
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

        print(f"  Loaded {len(texts)} texts for {emotion_name}. Classifying...")

        y_true = [emotion_name.lower()] * len(texts)
        y_pred = classify_texts(texts, classifier)

        if len(y_pred) != len(y_true):
             print(f"  Error: Number of predictions ({len(y_pred)}) does not match number of texts ({len(y_true)}). Skipping metrics calculation.")
             all_results[emotion_name] = {"error": "Prediction count mismatch."}
             continue

        # Determine all unique labels present in true and predicted lists for calculation
        present_labels = sorted(list(set(y_true + y_pred)))

        # Calculate metrics for all present labels
        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=present_labels, average=None, zero_division=0
        )

        # Store metrics specifically for the target emotion
        try:
            # Find the index corresponding to the target emotion within the 'present_labels' list
            target_idx = present_labels.index(emotion_name.lower())
            metrics = {
                "precision": float(p[target_idx]),
                "recall":    float(r[target_idx]),
                "f1":        float(f1[target_idx]),
                "support":   int(sup[target_idx]), # Support here refers to true instances of the target label
                #"predicted_support": y_pred.count(emotion_name.lower()) # Optionally add how many were predicted as target
            }
            all_results[emotion_name] = metrics
            print(f"  Metrics for {emotion_name}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, Support={metrics['support']}")

        except ValueError:
            # This happens if the target emotion was never predicted AND was not the true label (latter shouldn't happen here)
             print(f"  Warning: Target emotion '{emotion_name.lower()}' not found among predicted labels.")
             # Find support for true label (should be len(texts))
             true_support = y_true.count(emotion_name.lower())
             all_results[emotion_name] = {
                 "precision": 0.0, # Cannot have true positives if never predicted
                 "recall": 0.0,    # Cannot have true positives if never predicted
                 "f1": 0.0,
                 "support": true_support, # Correctly report the number of true samples
                 "info": "Target emotion was not predicted by the classifier for any sample."
             }


    print(f"\nSaving consolidated evaluation metrics to {args.output_metrics}...")
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    try:
        with open(args.output_metrics, "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print("✔ Successfully saved evaluation metrics.")
    except Exception as e:
         print(f"Error saving metrics file: {e}")

# --- Print Summary Table to Console ---
    print("\n=== Evaluation Summary (Console Output) ===")
    # Determine header length based on longest emotion name + padding
    max_emotion_len = max(len(e) for e in TARGET_10_LABELS) if TARGET_10_LABELS else 15
    header_format = f"{{:<{max_emotion_len}}} {{:<10}} {{:<10}} {{:<10}} {{:<10}}"
    row_format = f"{{:<{max_emotion_len}}} {{:<10}} {{:<10}} {{:<10}} {{:<10}}"

    print(header_format.format("Emotion", "Precision", "Recall", "F1-Score", "Support"))
    print("-" * (max_emotion_len + 43)) # Adjust line length dynamically

    for emotion, metrics in all_results.items():
        if "error" in metrics:
            # Ensure error message doesn't overflow too much
            error_msg = metrics['error'][:40] + '...' if len(metrics.get('error','')) > 40 else metrics.get('error','')
            print(f"{emotion:<{max_emotion_len}} Error: {error_msg}")
        else:
            # Extract metrics, providing 'N/A' if a key is missing
            precision = metrics.get('precision', 'N/A')
            recall = metrics.get('recall', 'N/A')
            f1 = metrics.get('f1', 'N/A')
            support = metrics.get('support', 'N/A')

            # Format numbers to 4 decimal places, keep 'N/A' as string
            p_str = f"{precision:.4f}" if isinstance(precision, (int, float)) else str(precision)
            r_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else str(recall)
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
            sup_str = str(support) # Support is usually an integer or N/A

            print(row_format.format(emotion, p_str, r_str, f1_str, sup_str))

    print("-" * (max_emotion_len + 43))
    # --- End Print Summary Table ---

if __name__ == "__main__":
    main()
