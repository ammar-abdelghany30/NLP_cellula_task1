import pandas as pd
import os
from datetime import datetime

DB_FILE = "classification_database.csv"

def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=[
            'timestamp', 'input_type', 'processed_text', 
            'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
        ])
        df.to_csv(DB_FILE, index=False)

def log_to_db(input_type: str, text: str, predictions: dict):
    init_db()
    new_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_type': input_type,
        'processed_text': text,
    }
    for label, info in predictions.items():
        new_entry[label] = f"{'TOXIC' if info['is_toxic'] else 'CLEAN'} ({info['probability']*100:.1f}%)"
        
    df = pd.read_csv(DB_FILE)
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(DB_FILE, index=False)

def get_all_records():
    init_db()
    return pd.read_csv(DB_FILE)