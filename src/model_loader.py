import torch
import torch.nn as nn
import pickle
import re

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optimal decision thresholds discovered during validation tuning
OPTIMAL_THRESHOLDS = {
    'toxic': 0.42,
    'severe_toxic': 0.31,
    'obscene': 0.49,
    'threat': 0.06,
    'insult': 0.45,
    'identity_hate': 0.15
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r'[^a-zA-Z0-9\s!\?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class BiLSTMToxicityModel(nn.Module):
    def __init__(self, vocab_size=25000, embedding_dim=250, hidden_dim=128, num_classes=6, dropout=0.3):
        super(BiLSTMToxicityModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled, _ = torch.max(lstm_out, dim=1)
        out = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(out)

def load_bilstm_pipeline(checkpoint_path='best_bilstm_no_glove.pth', vocab_path='vocab.pkl'):
    with open(vocab_path, 'rb') as f:
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
        results[col] = {
            'probability': round(prob, 4),
            'is_toxic': bool(prob >= thresh)
        }
    return results