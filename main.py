import os
import json
import re
import numpy as np
from datasets import load_dataset
from train import TransformerModel

# === Config (must match training) ===
block_size = 32
n_embd = 128
n_head = 4
n_layer = 4
model_path = "model_arj.json"
vocab_path = "vocab.json"

# === Step 1: Load or Rebuild Vocabulary ===
if os.path.exists(vocab_path):
    print("📦 Loading existing vocabulary...")
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)
        vocab = vocab_data["vocab"]
        # 🧠 Rebuild idx2word from vocab
        vocab_size = len(vocab)
        idx2word = [""] * vocab_size
        for word, idx in vocab.items():
            if idx < vocab_size:
                idx2word[idx] = word

else:
    print("📚 Rebuilding vocabulary...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = " ".join(x["text"] for x in ds if x["text"]).lower()
    tokens = re.findall(r"\b\w+\b", text)

    vocab = {"<unk>": 0}
    idx2word = ["<unk>"]
    encoded = []

    for word in tokens:
        if word not in vocab:
            vocab[word] = len(vocab)
            idx2word.append(word)
        encoded.append(vocab[word])

    vocab_size = len(vocab)
    with open(vocab_path, "w") as f:
        json.dump({"vocab": vocab, "idx2word": idx2word}, f)
    print("✅ Saved vocabulary to vocab.json")

print(f"✅ Vocab size: {vocab_size}")

# === Step 2: Load Model ===
model = TransformerModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer
)

with open(model_path, "r") as f:
    state = json.load(f)

for param, saved in zip(model.parameters(), state):
    param.data = np.array(saved["data"], dtype=np.float32)
    param.requires_grad = saved["requires_grad"]

print("✅ Loaded model weights.")

# === Step 3: Tokenization Helpers ===
def tokenize(text):
    return [vocab.get(w, 0) for w in re.findall(r"\b\w+\b", text.lower())]

def detokenize(token_ids):
    return " ".join(idx2word[i] if i < len(idx2word) else "<unk>" for i in token_ids)


# === Step 4: Inference Loop ===
print("\n🔍 Inference ready!")
while True:
    prompt = input("📝 Enter prompt (or empty to quit): ").strip()
    if not prompt:
        break

    prompt_ids = tokenize(prompt)[-block_size:]  # trim to last N tokens
    out_ids = model.generate(prompt_ids, max_new=30, top_k=10, temperature=0.8)
    generated_text = detokenize(out_ids)

    print("\n📤 Output:")
    print(generated_text)
    print("-" * 40)
