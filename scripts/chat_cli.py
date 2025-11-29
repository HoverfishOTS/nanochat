"""
Production Chat Interface (Clean Output)
"""
import argparse
import torch
import os
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.engine import Engine

# --- CONFIG ---
BOT_PERSONA = "khristian" 
# ----------------

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--tokenizer_path', type=str, required=True)
parser.add_argument('--temperature', type=float, default=0.8) 
parser.add_argument('--top_k', type=int, default=50)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Load Resources
tokenizer = RustBPETokenizer.from_directory(args.tokenizer_path)
checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

if "model_config" in checkpoint:
    config_args = checkpoint["model_config"]
    state_dict = checkpoint["model"]
else:
    print("Warning: No config found. Using Depth 8 fallback...")
    config_args = dict(
        n_layer=8,     # Change to 8 for baby model
        n_head=4,       # Change to 4 for baby model
        n_kv_head=4,    # Change to 4 for baby model
        n_embd=512,     # Change to 512 for baby model
        vocab_size=tokenizer.get_vocab_size()
    )
    state_dict = checkpoint

model = GPT(GPTConfig(**config_args))
model.load_state_dict(state_dict)
model.to(device)
model.eval()
engine = Engine(model, tokenizer)

print(f"\n[Ready] Chatting with {BOT_PERSONA}...")
print("="*40)

history = ""

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
        
    if user_input.lower() in ['quit', 'exit']: break
    if user_input.lower() == 'clear': 
        history = ""
        print("Memory wiped.")
        continue

    # Prompt Construction (Hidden from user)
    # We still feed the name to the model so it knows who to be,
    # but we won't print it to the screen.
    new_interaction = f"Unknown: {user_input}\n{BOT_PERSONA}: "
    full_prompt = history + new_interaction
    input_ids = tokenizer.encode(full_prompt)
    
    # Safety crop
    if len(input_ids) > 512:
        input_ids = input_ids[-512:]
    
    # No print here! The bot just starts talking.
    
    response_text = ""
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            gen = engine.generate(input_ids, num_samples=1, max_tokens=100, temperature=args.temperature, top_k=args.top_k)
            
            for token_col, _ in gen:
                token = token_col[0]
                word = tokenizer.decode([token])
                
                # CONSTRAINT: Stop at Newline
                if "\n" in word:
                    break
                
                print(word, end="", flush=True)
                response_text += word

    # Add to history so it remembers
    history += f"Unknown: {user_input}\n{BOT_PERSONA}: {response_text}\n"
    print()