"""
Debug Chat Interface - No Constraints
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
    # Fallback config
    config_args = dict(n_layer=8, n_head=4, n_kv_head=4, n_embd=512, vocab_size=tokenizer.get_vocab_size())
    state_dict = checkpoint

model = GPT(GPTConfig(**config_args))
model.load_state_dict(state_dict)
model.to(device)
model.eval()
engine = Engine(model, tokenizer)

print(f"\n[DEBUG MODE] Chatting with: {BOT_PERSONA}")
print("Constraints removed. The bot may speak for both sides.")
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

    # Prompt Construction
    new_interaction = f"Unknown: {user_input}\n{BOT_PERSONA}: "
    full_prompt = history + new_interaction
    input_ids = tokenizer.encode(full_prompt)
    
    # Safety crop (keep last 512 tokens of context)
    if len(input_ids) > 512:
        input_ids = input_ids[-512:]
    
    print(f"{BOT_PERSONA}: ", end="", flush=True)
    
    response_text = ""
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            # Increased max_tokens to 256 to let it ramble
            gen = engine.generate(input_ids, num_samples=1, max_tokens=256, temperature=args.temperature, top_k=args.top_k)
            
            for token_col, _ in gen:
                token = token_col[0]
                word = tokenizer.decode([token])
                
                # CONSTRAINT REMOVED: We no longer break on "\n"
                # The bot will likely generate "Unknown: ..." next
                
                print(word, end="", flush=True)
                response_text += word

    # Add the entire hallucinated sequence to history
    history += f"Unknown: {user_input}\n{BOT_PERSONA}: {response_text}\n"
    