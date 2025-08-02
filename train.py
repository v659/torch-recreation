from datasets import load_dataset
from arj_torch.tensor import ArjTensor
from arj_torch.nn import Embedding, Linear, LayerNorm, CrossEntropyLoss
from arj_torch.transformer import PatternTransformer
from arj_torch.optimizer import SGD
import json, re, os
import numpy as np
import time
import subprocess
import atexit
from colorama import Fore, Style

# === Setup ===
os.environ["OMP_NUM_THREADS"] = "1"
np.seterr(all='raise')

start_time = time.time()
caffeinate_proc = subprocess.Popen(['caffeinate', '-di'])
atexit.register(caffeinate_proc.terminate)

# === Config ===
block_size = 32
batch_size = 64
n_embd = 128
n_head = 4
n_layer = 4
lr = 3e-4
steps = 2000
preview_every = 1
warmup_steps = 100
save_path = "model_arj.json"


def pretty_log(step, loss, ema, lr, batch_size, steps):
    elapsed = time.time() - start_time
    per_step = elapsed / step
    eta = per_step * (steps - step)
    print(
        f"{Fore.RED}Step {step:04d}/{steps} "
        f"{Fore.YELLOW}Loss: {loss:.4f} "
        f"{Fore.BLUE}EMA: {ema:.4f} "
        f"{Fore.CYAN}LR: {lr:.6f} "
        f"{Fore.GREEN}Batch: {batch_size} "
        f"{Fore.MAGENTA}ETA: {eta / 60:.1f} min{Style.RESET_ALL}"
    )


def top_k_logits(logits, k):
    logits = np.array(logits)
    top_k_mask = logits < np.partition(logits, -k, axis=1)[:, -k][:, None]
    logits[top_k_mask] = -np.inf
    return logits


def top_p_logits(logits, p):
    logits = np.array(logits)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    sorted_indices = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
    cum_probs = np.cumsum(sorted_probs, axis=1)
    mask = cum_probs > p
    mask[:, 0] = False
    cutoff_indices = np.take_along_axis(sorted_indices, mask, axis=1)
    batch_indices = np.repeat(np.arange(logits.shape[0]), cutoff_indices.shape[1])
    logits[batch_indices, cutoff_indices.flatten()] = -np.inf
    return logits


# === Model ===
class TransformerModel:
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer):
        self.token_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Embedding(block_size, n_embd)
        self.transformer = PatternTransformer(seq_len=block_size, embed_dim=n_embd, heads=n_head, layers=n_layer)
        self.ln = LayerNorm(n_embd)
        self.head = Linear(n_embd, vocab_size)

    def __call__(self, x: ArjTensor):
        B, T = len(x.data), len(x.data[0])
        pos_ids = np.tile(np.arange(T), (B, 1))
        pos_ids = np.clip(pos_ids, 0, self.pos_emb.weight.shape[0] - 1)
        tok = self.token_emb(x)
        pos = self.pos_emb(ArjTensor(pos_ids))
        x = tok + pos
        x = self.transformer(x)
        x = self.ln(x)
        out = self.head(x)
        return out

    def parameters(self):
        return (
            self.token_emb.parameters() +
            self.pos_emb.parameters() +
            self.transformer.parameters() +
            self.ln.parameters() +
            self.head.parameters()
        )

    def generate(self, prompt_ids, max_new=30, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.2):
        x = ArjTensor([prompt_ids], requires_grad=False)
        generated = list(prompt_ids)
        for _ in range(max_new):
            logits = self(x)
            last = logits.data[0][-1]

            for token_id in generated:
                last[int(token_id)] /= repetition_penalty

            if top_k:
                last = top_k_logits([last], top_k)[0]
            if top_p:
                last = top_p_logits([last], top_p)[0]

            logits_np = np.array(last) / temperature
            exp_logits = np.exp(logits_np - np.max(logits_np))
            probs = exp_logits / np.sum(exp_logits)
            next_id = np.random.choice(len(probs), p=probs)

            generated.append(next_id)
            x.data = [x.data[0] + [next_id]]
        return generated


if __name__ == "__main__":
    print("Loading dataset...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = " ".join(x["text"] for x in ds if x["text"]).lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)

    vocab_path = "vocab.json"

    # === Load or rebuild vocab ===
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path) as f:
                vocab_raw = json.load(f)
                vocab = vocab_raw["vocab"]
            print("✅ Loaded existing vocab.")
        except Exception as e:
            print(f"⚠️ Failed to load vocab.json: {e}")
            vocab = None
    else:
        vocab = None

    if vocab is None:
        print("🔁 Rebuilding vocab from dataset...")
        vocab = {"<unk>": 0}
        idx2word = ["<unk>"]
        for word in tokens:
            if word not in vocab:
                vocab[word] = len(vocab)
                idx2word.append(word)
        with open(vocab_path, "w") as f:
            json.dump({"vocab": vocab}, f)
        print(f"✅ Saved vocab.json with {len(vocab)} tokens.")
    else:
        vocab_size = len(vocab)
        idx2word = [""] * vocab_size
        for word, idx in vocab.items():
            if idx < len(idx2word):
                idx2word[idx] = word

    encoded = [vocab.get(word, 0) for word in tokens]  # Use 0 for <unk>
    unk_count = encoded.count(0)
    if unk_count > 0:
        print(f"⚠️ {unk_count} unknown words mapped to <unk>")

    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    def get_batch():
        starts = np.random.randint(0, len(encoded) - block_size - 1, size=batch_size)
        x = np.array([encoded[start:start + block_size] for start in starts])
        y = np.array([encoded[start + 1:start + 1 + block_size] for start in starts])
        return ArjTensor(x, requires_grad=True), ArjTensor(y)

    model = TransformerModel(vocab_size, block_size, n_embd, n_head, n_layer)

    try:
        with open(save_path, "r") as f:
            loaded_state = json.load(f)
        for p, saved in zip(model.parameters(), loaded_state):
            p.data = np.array(saved["data"], dtype=np.float32)
            p.requires_grad = saved["requires_grad"]
        print("✅ Loaded saved model.")
    except FileNotFoundError:
        print("⚠️ No saved model found, starting fresh.")

    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, use_adam=True)

    print("🚀 Training started...")
    best_loss = float("inf")
    ema_loss = None

    for step in range(1, steps + 1):
        step_start = time.time()

        # Learning rate warmup
        if step <= warmup_steps:
            optimizer.set_lr(lr * step / warmup_steps)

        xb, yb = get_batch()
        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_time = time.time() - step_start
        loss_val = loss.data
        ema_loss = loss_val if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_val

        if step % 10 == 0:
            pretty_log(step, loss_val, ema_loss, optimizer.lr, batch_size, steps)
            print(f"⏱️ Step time: {step_time:.3f} sec")

        if step % preview_every == 0:
            prompt = xb.data[0][:8]
            out_ids = model.generate(prompt, max_new=20, top_k=10, temperature=0.8)
            print("Prompt:     ", " ".join(idx2word[int(i)] for i in prompt))
            print("Generated:  ", " ".join(idx2word[int(i)] for i in out_ids))
            print()

        if loss_val < best_loss and step % 50 == 0:
            best_loss = loss_val
            state = [{
                'data': p.data.tolist() if hasattr(p.data, 'tolist') else p.data,
                'requires_grad': p.requires_grad
            } for p in model.parameters()]
            with open(save_path, "w") as f:
                json.dump(state, f)
            print(f"✅ Saved best model (loss {best_loss:.4f})")

    print("🎉 Training complete.")
