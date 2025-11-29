import json
import os
import random
import pandas as pd # You might need: uv pip install pandas pyarrow
from pathlib import Path

# --- CONFIG ---
# For Monke, use the provided channel export JSON
# input_file = 'channel_export.json'
# output_dir = 'data/discord'        

# For AW, use below
input_file = "data/am_woman/am_woman.json"
output_dir = "data/am_woman"
val_split = 0.1

def prepare():
    # 1. Load Discord Data
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = data.get('messages', [])
    messages.sort(key=lambda x: x['timestamp'])

    # 2. Flatten to Text
    # NanoChat's pre-training just wants raw text to learn the "patterns"
    text_lines = []
    for msg in messages:
        author = msg.get('author', {}).get('nickname') or msg.get('author', {}).get('name', 'Unknown')
        content = msg.get('content', '')
        if content:
            # We format it so the model learns the structure of a chat
            text_lines.append(f"{author}: {content}\n")

    full_text = "".join(text_lines)
    print(f"Total characters: {len(full_text)}")

    # 3. Create Shards (Parquet)
    # NanoChat expects data to be sharded. For a small dataset, 1 shard is fine.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # We put the data into a Pandas DataFrame, then save as Parquet
    df = pd.DataFrame({'text': [full_text]})
    
    # Split for Train/Val (Simple method: Just save two files)
    # Realistically, for huge data, you'd split the text. 
    # For Discord logs, we'll just duplicate for simplicity or split lines.
    split_idx = int(len(text_lines) * (1 - val_split))
    train_text = "".join(text_lines[:split_idx])
    val_text = "".join(text_lines[split_idx:])

    print("Saving train.parquet...")
    pd.DataFrame({'text': [train_text]}).to_parquet(f'{output_dir}/train.parquet')
    
    print("Saving val.parquet...")
    pd.DataFrame({'text': [val_text]}).to_parquet(f'{output_dir}/val.parquet')

    print(f"Done! Data saved to {output_dir}")

if __name__ == '__main__':
    prepare()