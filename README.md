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

## Project Structure

