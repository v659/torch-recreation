from arj_torch.tensor import ArjTensor
from arj_torch.nn import Embedding, Linear, ReLU, MSELoss, SGD, Sequential

# === Example Data ===
# Each input is a pair of token IDs (integers)
inputs = ArjTensor([[1, 3], [4, 2]], requires_grad=True)  # shape (2, 2)
targets = ArjTensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)  # desired outputs

# === Model ===
model = Sequential(
    Embedding(vocab_size=10, embedding_dim=4),  # output: (2, 8)
    Linear(in_features=8, out_features=2),      # output: (2, 2)
    ReLU()
)

# === Loss + Optimizer ===
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)

# === Training ===
for epoch in range(30):
    # Forward pass
    pred = model(inputs)
    loss = loss_fn(pred, targets)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}: Loss = {loss}")
    print(f"prediction: {pred}")
