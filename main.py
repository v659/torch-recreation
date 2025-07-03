from arj_torch.nn import *
import matplotlib.pyplot as plt

# === Input & Targets ===
# 3 samples, 2 token IDs each
inputs = ArjTensor([[1, 3], [4, 2], [1, 4]], requires_grad=True)
targets = ArjTensor([[0], [1], [0]])  # integer labels for CrossEntropyLoss

# === Model ===
model = Sequential(
    Embedding(vocab_size=10, embedding_dim=4),  # (3, 2) → (3, 8)
    Linear(in_features=8, out_features=2)       # (3, 8) → (3, 2)
)

# === Loss & Optimizer ===
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1)

# === Training Loop ===
loss_values = []
accuracy_values = []

for epoch in range(50):
    # Forward pass
    logits = model(inputs)
    loss = loss_fn(logits, targets)

    # Backward pass
    loss.backward()

    # Parameter update
    optimizer.step()
    optimizer.zero_grad()

    # Accuracy calculation
    pred_classes = [max(enumerate(row), key=lambda x: x[1])[0] for row in logits.data]
    true_classes = [row[0] if isinstance(row, list) else row for row in targets.data]
    accuracy = sum([p == t for p, t in zip(pred_classes, true_classes)]) / len(targets.data)

    # Logging
    loss_values.append(loss.data)
    accuracy_values.append(accuracy)
    print(f"Epoch {epoch + 1:2d}: Loss = {loss.data:.4f}, Accuracy = {accuracy:.2f}")

# === Plotting ===
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_values, marker='o', label='Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accuracy_values, marker='s', color='green', label='Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.grid(True)

plt.tight_layout()
plt.show()
