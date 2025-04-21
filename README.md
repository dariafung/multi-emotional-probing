# Multi-Emotional Probing for Controllable Language Generation: Exploring the Geometry of Emotion in Transformer Models

**Team Members:**

- Daria Feng (yfeng266@wisc.edu) - Computer Science  
- Eric Xu (zxu684@wisc.edu) - Statistics  
- Shengbo Qian (sqian37@wisc.edu) - Statistics  
- Huiyu Li (hli798@wisc.edu) - Statistics

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Key Features](#key-features)  
- [Project Structure](#project-structure)  
- [Setup & Installation](#setup--installation)  
- [Dataset](#dataset)  
- [Workflow & Usage](#workflow--usage)  
  - [1. Data Preprocessing](#1-data-preprocessing)  
  - [2. Part A: BERT Emotion Probing](#2-part-a-bert-emotion-probing)  
  - [3. Part B: Visualization](#3-part-b-visualization)  
  - [4. Part C: GPT-2 Emotion Injection & Generation](#4-part-c-gpt-2-emotion-injection--generation)  
  - [5. Text Evaluation](#5-text-evaluation)  
- [Example Results](#example-results)  
- [Future Work / Limitations](#future-work--limitations)  
- [References](#references)  
- [License](#license)  

---

## Project Overview

Large language models (LLMs) excel at generating human-like text, but controlling their emotional expression remains a challenge. Current methods like prompt engineering offer coarse control, while fine-tuning is resource-intensive. This project aims to understand how emotions are internally represented in Transformer models (specifically BERT) and develop a method to directly manipulate these representations for fine-grained emotional control in text generation (using GPT-2).

We employ a two-stage approach:
1. **Probing:** Train linear probes on BERT's hidden states to identify "emotion direction vectors" in its latent space using the GoEmotions dataset.  
2. **Injection & Generation:** Inject these extracted vectors (or linear combinations for mixing emotions) into the hidden layers of a generative model (GPT-2) to control the emotional tone of the generated text.

This allows for exploring the geometry of emotion within models and enables controllable, multi-emotional text generation.

---

## Key Features

- **Emotion Probing**: Trains linear probes on BERT layers to classify emotions from hidden states.  
- **Vector Extraction**: Extracts weight vectors representing specific emotions in latent space.  
- **Visualization**: PCA-based visualization of emotion vector geometry.  
- **Controllable Generation**: Inject emotion vectors into GPT-2 to control output tone.  
- **Emotion Mixing**: Combine multiple vectors for mixed-emotion generation.  
- **Evaluation**: Scripts for evaluating both probe accuracy and generation quality.

---

# Emotion Control Project

## Project Structure

```
emotion_control_project/
├── data/
│   ├── processed/
│   └── preprocess.py
├── probe_model/
│   ├── checkpoints/
│   ├── vectors/
│   ├── run_probe.py
│   ├── eval_metrics.py
│   ├── extract_vectors.py
│   └── metrics.csv
├── visualization/
│   ├── plot_pca.py
│   └── heatmap.py
├── generation/
│   ├── gpt2_inject.py
│   └── eval_text.py
├── 01_probe_small.ipynb
├── 02_generation_small.ipynb
├── results/
│   ├── generation_output/
│   ├── evaluation_results/
│   └── all_emotions_eval.json
└── README.md
```

## Setup & Installation

### 1. Prerequisites
- Python 3.8+
- `pip` or `conda`

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd emotion_control_project
```

### 3. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 4. Install Dependencies
Create requirements.txt:

```txt
torch
transformers
datasets
pandas
scikit-learn
matplotlib
emoji
tqdm
pyarrow
```

Install:

```bash
pip install -r requirements.txt
```

## Dataset
We use the GoEmotions dataset by Google Research (~58k Reddit comments labeled with 27 emotions + neutral).
We focus on the following 10:
neutral, admiration, gratitude, curiosity, love, anger, joy, sadness, amusement, confusion

Prepare the dataset:

```bash
python data/preprocess.py --out_dir data/processed
```

## Workflow & Usage

### 1. Data Preprocessing
```bash
python data/preprocess.py --out_dir data/processed
```

### 2. Part A: BERT Emotion Probing

#### a) Train Linear Probes
```bash
python probe_model/run_probe.py \
  --data_dir data/processed \
  --bert_name bert-large-uncased \
  --layers 0 6 10 13 15 17 20 23 \
  --epochs 3 \
  --batch_size 32 \
  --lr 1e-3
```

#### b) Extract Emotion Vectors
```bash
python probe_model/extract_vectors.py
```

#### c) Plot Probe F1 Scores
```bash
python probe_model/eval_metrics.py
```

### 3. Part B: Visualization (PCA)
```bash
python visualization/plot_pca.py
```

### 4. Part C: GPT-2 Emotion Injection & Generation

#### a) Generate with a Single Emotion
```bash
python generation/gpt2_inject.py \
  --model_name gpt2-medium \
  --prompt "Describe a time you felt unfairly treated." \
  --vector_matrix probe_model/vectors/layer_20.pt \
  --emotion_name anger \
  --layer 10 \
  --output_file results/generated_anger_example.txt \
  --max_length 150 \
  --temperature 0.8
```

#### b) Generate with Mixed Emotions
```bash
python generation/emotion_mix.py \
  --model_name gpt2-medium \
  --prompt "The news was unexpected and difficult to hear." \
  --vector_matrices probe_model/vectors/layer_20.pt probe_model/vectors/layer_20.pt \
  --emotion_names anger sadness \
  --weights 0.7 0.3 \
  --layer 10 \
  --output_file results/mixed_anger_sadness_example.txt \
  --max_length 150 \
  --temperature 0.8
```

#### c) Batch Generation
Use a script or loop to generate samples for all 10 emotions and save under:
```bash
results/generation_output/generated_<emotion>/
```

### 5. Text Evaluation
```bash
python generation/eval_text.py \
  --input_dir results/generation_output \
  --output_metrics results/all_emotions_eval.json \
  --model_name bhadresh-savani/bert-base-uncased-emotion
```

## Example Results
- Probe Accuracy: probe_model/metrics.csv
- PCA Plots: visualization/pca_layer_X.png
- Generated Text: results/generation_output/
- Evaluation Metrics: results/all_emotions_eval.json

## Future Work / Limitations
- LLaMA-2 Integration: Not yet implemented
- GPT-2 Limitations: Lower fluency than SOTA LLMs
- Nonlinear Probes: Consider using MLP for richer emotion capture
- Alternate Injection: Beyond simple vector addition
- Human Evaluation: Needed for better emotional nuance
- Vector Purity: Improve disentanglement between emotions

## References
- LATENTQA: Teaching LLMs to Decode Activations into Natural Language, arXiv:2412.08686 (2024)
- Towards Explainable Multimodal LLMs: A Comprehensive Survey, arXiv:2412.02104 (2024)
- LM-Debugger: Inspection & Intervention in Transformers, arXiv:2204.12130 (2022)
- The Better Your Syntax, the Better Your Semantics?, EMNLP 2022
- Mechanistic Interpretability of Emotion Inference in LLMs, arXiv:2502.05489 (2024)

## License

```
MIT License

Copyright (c) 2025 Daria Feng, Eric Xu, Shengbo Qian, Huiyu Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

