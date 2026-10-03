import torch
import torch.nn as nn
import pickle
import re
from collections import Counter

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimal decision thresholds discovered during validation tuning
OPTIMAL_THRESHOLDS = {
    "toxic": 0.42,
    "severe_toxic": 0.31,
    "obscene": 0.49,
    "threat": 0.06,
    "insult": 0.45,
    "identity_hate": 0.15,
}


# --- 1. Vocabulary Class Definition (Required for Pickle Unpickling) ---
class Vocabulary:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)

        most_common = word_counts.most_common(self.max_size - 2)
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=35):
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(w, self.word2idx[self.unk_token]) for w in tokens]
        if len(indices) < max_len:
            indices = indices + [self.word2idx[self.pad_token]] * (
                max_len - len(indices)
            )
        else:
            indices = indices[:max_len]
        return indices


# --- 2. Text Preprocessing Function ---
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"[^a-zA-Z0-9\s!\?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- 3. BiLSTM Neural Network Architecture ---
class BiLSTMToxicityModel(nn.Module):
    def __init__(
        self,
        vocab_size=25000,
        embedding_dim=250,
        hidden_dim=128,
        num_classes=6,
        dropout=0.3,
    ):
        super(BiLSTMToxicityModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled, _ = torch.max(lstm_out, dim=1)
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits


# --- 4. Pipeline Load & Prediction Functions ---
import os


def load_bilstm_pipeline(
    checkpoint_name="best_bilstm_LargeSet.pth", vocab_name="vocab.pkl"
):
    # Determine directory containing model_loader.py
    base_dir = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = os.path.join(base_dir, checkpoint_name)
    vocab_path = os.path.join(base_dir, vocab_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found at: {vocab_path}")

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    model = BiLSTMToxicityModel(vocab_size=len(vocab.word2idx)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, vocab


def predict_toxicity(text: str, model, vocab) -> dict:
    cleaned = clean_text(text)
    encoded = vocab.encode(cleaned, max_len=150)
    tensor_input = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = {}
    for idx, col in enumerate(label_cols):
        prob = float(probs[idx])
        thresh = OPTIMAL_THRESHOLDS[col]
        results[col] = {"probability": round(prob, 4), "is_toxic": bool(prob >= thresh)}
    return results
