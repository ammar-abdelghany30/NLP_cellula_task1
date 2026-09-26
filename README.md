# 🛡️ Multimodal Content Moderation with Bidirectional LSTM

A PyTorch-based Deep Learning model designed to detect safety violations and toxic categories in multimodal user interactions. By evaluating combined text sequences containing both **user queries** and **image descriptions**, the model achieves robust multi-class toxicity classification across 9 safety categories.

---

## 📌 Project Overview

Modern AI assistants (e.g., ChatGPT with Vision, Gemini) evaluate safety based on the **interaction context** between text prompts and accompanying visual scenes. Toxicity often exists not in isolation, but in the combination of inputs:
- An innocent query (*"Real estate advice"*) paired with a dangerous image (*"Crime scene with police tape"*) indicates a **Violent Crime** context.
- A benign image (*"Family picnic"*) paired with a cyberattack query (*"DDoS botnet guide"*) indicates an **Unknown S-Type / Cyber Threat**.

This repository implements a **Bidirectional LSTM (BiLSTM)** architecture trained to classify these interactions and enforce safety standards.

---

## 🚀 Key Results & Performance

- **Target Metric**: Weighted F1 Score $\ge 0.75$
- **Achieved Test Weighted F1**: **`0.9465` (94.65%)**
- **Test Accuracy**: **`94.89%`** (427 / 450 test samples correctly classified)

### Classification Report Summary

| Toxicity Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Child Sexual Exploitation** | 1.00 | 1.00 | **1.00** | 15 |
| **Elections** | 1.00 | 1.00 | **1.00** | 17 |
| **Non-Violent Crimes** | 0.94 | 0.98 | **0.96** | 45 |
| **Safe** | 0.89 | 0.99 | **0.94** | 150 |
| **Sex-Related Crimes** | 1.00 | 0.94 | **0.97** | 17 |
| **Suicide & Self-Harm** | 1.00 | 1.00 | **1.00** | 17 |
| **Unknown S-Type** | 0.94 | 0.59 | **0.72** | 29 |
| **Violent Crimes** | 0.99 | 0.94 | **0.97** | 119 |
| **Unsafe** | 1.00 | 1.00 | **1.00** | 41 |

---

## ⚙️ How It Works

1. **Feature Concatenation (`sep` boundary)**:
   The model concatenates the user query and image description using an explicit boundary token:
   $$\text{Input} = \text{Query} + \text{" sep "} + \text{Image Description}$$
   This structural anchor prevents modal confusion and allows the network to process prompt intent and visual context separately.

2. **Custom Tokenization & Padding**:
   Texts are tokenized using a custom `Vocabulary` class (`max_size=5000`) and padded/truncated to a sequence length of `36` tokens, capturing $>95\%$ of sample lengths.

3. **BiLSTM Network Architecture**:
   - **Embedding Layer**: Converts token IDs into dense 128-dimensional vectors.
   - **2-Layer Bidirectional LSTM**: Processes sequences forward and backward to extract temporal/semantic context.
   - **Global Max Pooling**: Extracts the strongest feature activations across the sequence.
   - **Dense Classification Head**: Fully connected layers with ReLU activation, Dropout ($p=0.3$), and 9-class Softmax outputs.

4. **Class Weighting & Early Stopping**:
   - Applied class weights to `nn.CrossEntropyLoss` to handle severe class imbalance (minority classes make up only ~3.4% of data).
   - Utilized **Early Stopping** based on validation loss to prevent overfitting and checkpoint optimal model weights.

---

## ⚖️ Pros & Cons

### ✅ Pros
- **High Benchmark Performance**: Achieves $>94\%$ F1 score across diverse safety categories.
- **Handles Class Imbalance**: Perfect recall ($1.00$) on critical minority categories (*Child Sexual Exploitation*, *Suicide & Self-Harm*).
- **Fast Execution**: Compact memory footprint (~5000 vocabulary size); trains and runs inference quickly on CPU/GPU.
- **Structural Boundary Awareness**: Inclusion of `sep` token significantly reduced false positives between `Unknown S-Type` and `Safe`.

### ❌ Cons / Limitations
- **Shortcut Learning (Spurious Correlations)**: Because embeddings were trained from scratch on 3,000 samples, the model sometimes relies on single keyword associations (e.g., assuming *"laptop"* always implies a *Non-Violent Crime*).
- **Lack of Deep Semantic Context**: Fails on subtle edge cases or reworded prompts outside the specific synthetic dataset distributions.
- **Lower Recall on `Unknown S-Type`**: Ambiguous queries paired with neutral images achieve lower recall ($0.59$), occasionally defaulting to `Safe`.

---

## 📁 Project Structure

```text
.
├── RNN/
│   ├── train_bilstm.ipynb      # Main training, EDA, and evaluation notebook
│   ├── inference_test.ipynb    # Inference script for testing dummy/unseen prompts
│   ├── bilstm_toxicity_model.pth # Saved PyTorch model weights
│   ├── vocab.pkl               # Saved custom Vocabulary dictionary
│   └── label_encoder.pkl       # Saved Scikit-Learn LabelEncoder instance
├── assets/
│   ├── learning_curves.png     # Train vs Val Loss & Validation F1 curves
│   └── confusion_matrix.png    # Test evaluation confusion matrix heatmap
├── cellula toxic data.csv      # Dataset file
├── Report.pdf                  # Full project storyline & evaluation report
├── LICENSE                     # MIT License
└── README.md                   # Project documentation