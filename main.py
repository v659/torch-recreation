from arj_torch.nn import *

# === Example Data ===
inputs = ArjTensor([[1, 3], [4, 2], [1, 4]], requires_grad=True)  # shape (3, 2)
targets = ArjTensor([[0], [1], [0]])  # class labels for CrossEntropyLoss (not one-hot)

# === Model ===
model = Sequential(
    Embedding(vocab_size=10, embedding_dim=4),  # output shape: (3, 8)
    Linear(in_features=8, out_features=2)       # output shape: (3, 2)
)

# === Loss + Optimizer ===
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1)

# === Training Loop ===
for epoch in range(50):
    # Forward pass
    logits = model(inputs)  # raw scores
    loss = loss_fn(logits, targets)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}: Loss = {loss.data}")
    print(f"Prediction (logits): {logits.data}")
