"""
Download a slice of FineWeb-Edu for pre-training.
"""
import os
import requests

# Config
DATA_DIR = "data/fineweb"
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"

def download_shards(n=50):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading {n} shards of FineWeb-Edu to {DATA_DIR}...")
    
    for i in range(n):
        filename = f"shard_{i:05d}.parquet"
        filepath = os.path.join(DATA_DIR, filename)
        url = f"{BASE_URL}/{filename}"
        
        if os.path.exists(filepath):
            print(f" - {filename} exists, skipping.")
            continue
            
        print(f" - Downloading {filename}...")
        try:
            r = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk: f.write(chunk)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    # 50 shards = ~5GB of data. 
    # Increase 'n' if you want more (up to 1822 shards total).
    download_shards(n=50)