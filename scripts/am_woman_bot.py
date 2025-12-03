"""
NanoChat Discord Bot - Deployment Ready
"""
import discord
import torch
import os
import argparse
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.engine import Engine

# --- CONFIGURATION ---
# 1. Discord Bot Token (Get this from Discord Developer Portal)
secret_path = os.path.join(os.path.dirname(__file__), '..', 'secret.txt')
with open(secret_path, 'r') as f:
    DISCORD_TOKEN = f.read().strip()

# 2. The exact name the bot should pretend to be (Must match training data!)
BOT_PERSONA = "khristian"

# 3. Paths to best model
TOKENIZER_PATH = "data/am_woman"
# Update this to latest checkpoint
MODEL_PATH = r"base_checkpoints\\am_woman_v2\\model_002000.pt"

# 4. Generation Settings
TEMPERATURE = 1.2  # Creativity
TOP_K = 50         # Stability
# ---------------------

# Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Brain on {device}...")

# Load Tokenizer
tokenizer = RustBPETokenizer.from_directory(TOKENIZER_PATH)

# Load Model
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

if "model_config" in checkpoint:
    config_args = checkpoint["model_config"]
    state_dict = checkpoint["model"]
else:
    print("Warning: No config found. Using Depth 12 fallback...")
    config_args = dict(
        n_layer=12,     # Change to 12 for updated model
        n_head=6,       # Change to 6 for updated model
        n_kv_head=6,    # Change to 6 for updated model
        n_embd=768,     # Change to 768 for updated model
        vocab_size=tokenizer.get_vocab_size()
    )
    state_dict = checkpoint

model = GPT(GPTConfig(**config_args))
model.load_state_dict(state_dict)
model.to(device)
model.eval()
engine = Engine(model, tokenizer)
print("Brain Loaded. Connecting to Discord...")

# Discord Client Setup
intents = discord.Intents.default()
intents.message_content = True # Critical: Allows bot to read chat
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    print("Bot is live! Ctrl+C to stop.")

@client.event
async def on_message(message):
    # 1. Ignore own messages
    if message.author == client.user:
        return

    # 2. (Optional) Only reply if mentioned, or in specific channels
    # if not client.user.mentioned_in(message): return
    # if message.channel.id != "1445544188874719273":
    #     return

    # 3. Construct Prompt
    # We format it exactly like the training data: "User: Message\nBot: "
    user_name = message.author.display_name # Uses server nickname if available
    clean_content = message.content.strip()
    
    if not clean_content:
        return

    prompt = f"{user_name}: {clean_content}\n{BOT_PERSONA}: "
    print(f"Reading: {prompt.strip()}")

    # 4. Tokenize
    input_ids = tokenizer.encode(prompt)
    # Safety crop to fit context window
    if len(input_ids) > 512:
        input_ids = input_ids[-512:]

    # 5. Generate Response
    response_text = ""
    async with message.channel.typing(): # Show "Bot is typing..."
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                gen = engine.generate(
                    input_ids, 
                    num_samples=1, 
                    max_tokens=100, 
                    temperature=TEMPERATURE, 
                    top_k=TOP_K
                )
                
                for token_col, _ in gen:
                    token = token_col[0]
                    word = tokenizer.decode([token])
                    
                    # STOP CONDITION: Newline means it's trying to impersonate someone else
                    # if "\n" in word:
                    #     break
                    
                    response_text += word

    # 6. Send
    if response_text.strip():
        print(f"Replying: {response_text}")
        await message.channel.send(response_text)

# Run the bot
if __name__ == "__main__":
    if DISCORD_TOKEN == "YOUR_TOKEN_HERE":
        print("Error: You must set your DISCORD_TOKEN in the script!")
    else:
        client.run(DISCORD_TOKEN)