from arj_torch.nn import *
model = Linear(1, 1)
optim = SGD(model.parameters(), lr=0.1)

# Fake data: input 2.0 → target 4.0
x = ArjTensor([[2.0]], requires_grad=True)
y = ArjTensor(4.0)

for epoch in range(10):
    pred = model(x)
    loss = (pred - y) * (pred - y)  # MSE
    loss.backward()

    optim.step()
    print(f"Epoch {epoch}: pred={pred}, loss={loss}")
