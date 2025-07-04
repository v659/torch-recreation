import json
from arj_torch.tensor import ArjTensor
from arj_torch.nn import Linear, ReLU, Sequential

# === Define the same model architecture as used during training ===
model = Sequential(
    Linear(in_features=4, out_features=32),
    ReLU(),
    Linear(in_features=32, out_features=1)
)

# === Load model parameters ===
def load_model(model, path):
    with open(path, 'r') as f:
        param_data = json.load(f)

    for p, saved in zip(model.parameters(), param_data):
        p.data = saved['data']
        p.requires_grad = saved['requires_grad']
    print(f"📥 Model loaded from {path}")

load_model(model, "pattern_model.json")  # Path to your saved model

# === Take user input ===
user_input = input("Enter 4 numbers separated by spaces: ")
try:
    numbers = list(map(float, user_input.strip().split()))
    if len(numbers) != 4:
        raise ValueError
except ValueError:
    print("❌ Please enter exactly 4 valid numbers.")
    exit()

# === Normalize input ===
norm_input = [[v / 100 for v in numbers]]  # shape: (1, 4)
input_tensor = ArjTensor(norm_input, requires_grad=False)

# === Make prediction ===
output = model(input_tensor)
prediction = output.data[0][0] * 100  # De-normalize

print(f"📊 Predicted next number: {round(prediction):.2f}")
