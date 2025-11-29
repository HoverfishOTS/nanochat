"""
Train a tokenizer on custom Discord data.
"""
import os
import time
import argparse
import glob
import torch
import pyarrow.parquet as pq
from nanochat.tokenizer import RustBPETokenizer

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on')
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document')
parser.add_argument('--vocab_size', type=int, default=4096, help='Vocabulary size') # Defaulted to 4096 for you
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# CUSTOM DISCORD TEXT ITERATOR
# This replaces the complex 'parquets_iter_batched' from the original repo

def text_iterator():
    # Hardcoded path to discord data
    data_dir = "data/am_woman"
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    
    if not files:
        print(f"CRITICAL ERROR: No .parquet files found in {data_dir}")
        print("Did you run the prepare_discord.py script?")
        return

    print(f"Found {len(files)} shard(s) in {data_dir}. Reading...")

    nchars = 0
    for filepath in files:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            
            for doc_text in texts:
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[:args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to data/am_woman
tokenizer_dir = "data/am_woman"
print(f"Saving tokenizer to {tokenizer_dir}...")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = "Hello world! This is a test."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"Sanity Check: '{test_text}' -> {encoded} -> '{decoded}'")
assert decoded == test_text

# -----------------------------------------------------------------------------
# Calculate and save token bytes (Required for training metrics)
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]
    if token_str in special_set:
        token_bytes.append(0)
    else:
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")