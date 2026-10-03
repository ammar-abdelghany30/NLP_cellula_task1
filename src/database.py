import pandas as pd
import os
from datetime import datetime

DB_FILE = "classification_database.csv"

def init_db():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, DB_FILE)
    
    if not os.path.exists(db_path):
        df = pd.DataFrame(columns=['timestamp', 'input_type', 'processed_text', 'output'])
        df.to_csv(db_path, index=False)
    return db_path

def log_to_db(input_type: str, text: str, pred_result: dict):
    db_path = init_db()
    
    new_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_type': input_type,
        'processed_text': text,
        'output': pred_result['summary']  # e.g., "TOXIC (toxic, insult)" or "CLEAN"
    }
    
    df = pd.read_csv(db_path)
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(db_path, index=False)

def get_all_records():
    db_path = init_db()
    return pd.read_csv(db_path)