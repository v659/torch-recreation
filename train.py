from arj_torch.tensor import ArjTensor
from arj_torch.nn import Linear, ReLU, LayerNorm, MSELoss, SGD, Sequential, Embedding
import random
import matplotlib.pyplot as plt
from arj_torch.transformer import PatternTransformer
import json


def save_model(model, path):
    params = []
    for p in model.parameters():
        params.append({
            'data': p.data,
            'requires_grad': p.requires_grad
        })
    with open(path, 'w') as f:
        json.dump(params, f)
    print(f"✅ Model saved to {path}")

# === Pattern Dataset ===
def generate_pattern_sample():
    kind = random.choice(['add', 'double', 'fib'])
    if kind == 'add':
        a = random.randint(1, 50)
        step = random.randint(1, 10)
        return [a, a + step, a + 2 * step, a + 3 * step], [a + 4 * step]
    elif kind == 'double':
        a = random.randint(1, 10)
        return [a, a * 2, a * 4, a * 8], [a * 16]
    elif kind == 'fib':
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        return [a, b, a + b, a + 2 * b], [2 * a + 3 * b]


def print_weight_equations(epoch, lr, param_count):
    print(f"\n[Epoch {epoch}] Weight Update Equations:")
    for i in range(param_count):
        print(f"W{i}_new = W{i}_old - {lr} * dL/dW{i}")

# === Training Data ===
train_inputs, train_targets = [], []
for _ in range(500):
    x, y = generate_pattern_sample()
    train_inputs.append([v / 100 for v in x])
    train_targets.append([y[0] / 100])

inputs = ArjTensor(train_inputs, requires_grad=True)
targets = ArjTensor(train_targets, requires_grad=True)

# === Test Sequences ===
test_samples = [
    [2, 4, 6, 8],
    [3, 6, 12, 24],
    [1, 2, 3, 4],
    [5, 10, 15, 20],
    [2, 3, 5, 8],
    [2, 4, 8, 16]
]
test_inputs = [[v / 100 for v in x] for x in test_samples]
test_tensor = ArjTensor(test_inputs, requires_grad=False)


# === Ground Truth (optional, for plotting)
def true_next_value(seq):
    if seq[1] - seq[0] == seq[2] - seq[1] == seq[3] - seq[2]:
        return seq[3] + (seq[1] - seq[0])
    elif seq[1] == seq[0] * 2 and seq[2] == seq[1] * 2 and seq[3] == seq[2] * 2:
        return seq[3] * 2
    elif seq[2] == seq[0] + seq[1] and seq[3] == seq[1] + seq[2]:
        return seq[2] + seq[3]
    else:
        return None


actual = [true_next_value(seq) for seq in test_samples]

# === Model ===
use_transformer = False
if use_transformer:
    print("Using Transformer")
    model = PatternTransformer()
else:
    print("No transformer")
    model = Sequential(
        Linear(in_features=4, out_features=32),
        ReLU(),
        Linear(in_features=32, out_features=1)
    )

loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01, use_adam=True)

# === Matplotlib Setup ===
fig, ax = plt.subplots(figsize=(10, 5))
labels = [str(s) for s in test_samples]
x = list(range(len(labels)))
bar_actual = ax.bar(x, actual, width=0.4, label='Actual', align='center')
bar_pred = ax.bar([i + 0.4 for i in x], [0] * len(x), width=0.4, label='Predicted', align='center', color='orange')
ax.set_xticks([i + 0.2 for i in x])
ax.set_xticklabels(labels, rotation=45)
ax.set_ylim(0, max(actual) * 1.5)
ax.set_ylabel("Next Value")
ax.set_title("Live Prediction vs Actual")
ax.legend()
plt.tight_layout()


# === Update Plot Function ===
def update_plot(epoch):
    pred = model(test_tensor)
    predicted = [p[0] * 100 for p in pred.data]
    for bar, height in zip(bar_pred, predicted):
        bar.set_height(height)
    ax.set_title(f"Live Prediction vs Actual (Epoch {epoch})")


# === Training + Animation Loop ===
loss_values = []
for epoch in range(1, 10001):
    pred = model(inputs)
    loss = loss_fn(pred, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_values.append(loss.data)

    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Loss = {loss.data:.6f}")
        pred = model(test_tensor)
        print_weight_equations(epoch + 1, lr = 0.01, param_count = len(model.parameters()))
        for seq, out, target in zip(test_samples, pred.data, actual):
            print(f"{seq} → predicted: {out[0] * 100:.2f} | actual: {target:.2f}")
        save_model(model, 'pattern_model.json')
        print("Model saved")
        update_plot(epoch)
        plt.pause(0.01)

# === Final Predictions ===
print("\n--- Final Predictions ---")
final_pred = model(test_tensor)
for seq, out, target in zip(test_samples, final_pred.data, actual):
    print(f"{seq} → predicted: {out[0] * 100:.2f} | actual: {target:.2f}")
save_model(model, 'pattern_model.json')
print("Model saved")
plt.show()



# === Final Loss Plot ===
plt.figure()
plt.plot(loss_values)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
