import random
import math

import numpy as np
from arj_torch.tensor import ArjTensor

class Linear:
    def __init__(self, in_features, out_features, use_he_init=True):
        self.in_features = in_features
        self.out_features = out_features
        if use_he_init:
            limit = math.sqrt(2 / in_features)  # He Initialization for ReLU
        else:
            limit = math.sqrt(6 / (in_features + out_features))  # Xavier Uniform

        self.weight = ArjTensor(
            np.random.uniform(-limit, limit, (in_features, out_features)).tolist(),
            requires_grad=True
        )
        # Initialize bias with small positive value to help ReLU "activate"
        self.bias = ArjTensor(np.full((1, out_features), 0.01), requires_grad=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input = x  # Save for backward
        x_np = np.array(x.data)
        while x_np.ndim > 3 and x_np.shape[1] == 1:
            x_np = x_np.squeeze(1)  # Squeeze only if dim==1

        w_np = np.array(self.weight.data)  # (in_features, out_features)
        b_np = np.array(self.bias.data)
        is_3d = (x_np.ndim == 3)
        if not is_3d:
            x_np = x_np[:, np.newaxis, :]  # (B, 1, in_features)

        out_np = np.matmul(x_np, w_np) + b_np.reshape(1, 1, -1)
        out_list = out_np.tolist()

        out = ArjTensor(out_list, requires_grad=True, _children=(x, self.weight, self.bias), _op="linear")

        def _backward():
            if not x.requires_grad:
                return

            grad_out = np.array(out.grad)  # (B, T, out_features)
            x_data = np.array(self.input.data)
            w_np_local = np.array(self.weight.data)  # ✅ define here

            if not is_3d:
                grad_out = grad_out[:, 0, :]
                x_data = x_data[:, np.newaxis, :]

            B, T = grad_out.shape[0], grad_out.shape[1]
            x_flat = x_data.reshape(-1, self.in_features)
            grad_out_flat = grad_out.reshape(-1, self.out_features)

            grad_w = np.matmul(grad_out_flat.T, x_flat)  # (out_features, in_features)
            grad_b = np.sum(grad_out, axis=(0, 1), keepdims=True)  # (1, out_features)

            self.weight.grad = grad_w.T.tolist()  # transpose to (in, out)
            self.bias.grad = grad_b.tolist()

            grad_input = np.matmul(grad_out, w_np_local.T)  # (B, T, in_features)
            if not is_3d:
                grad_input = grad_input[:, 0, :]
            x.grad = grad_input.tolist()

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight, self.bias]




class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = ArjTensor([[random.uniform(-1, 1) for _ in range(embedding_dim)] for _ in range(vocab_size)], requires_grad=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x_np = np.array(x.data, dtype=int)
        x_np = np.clip(x_np, 0, self.weight.shape[0] - 1)
        weight_np = np.array(self.weight.data)
        embedded = weight_np[x_np]  # Fancy indexing
        return ArjTensor(embedded.tolist(), requires_grad=True, _children=(self.weight,), _op="embedding")

    def parameters(self):
        return [self.weight]


class ReLU:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x_np = np.array(x.data)
        relu_out = np.maximum(0, x_np)

        out = ArjTensor(relu_out.tolist(), requires_grad=True, _children=(x,), _op="relu")

        def _backward():
            if not x.requires_grad:
                return
            grad_out = np.array(out.grad)
            grad_input = grad_out * (x_np > 0).astype(np.float32)
            x.grad = grad_input.tolist()

        out._backward = _backward
        return out



class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.dim = dim
        self.gamma = ArjTensor([[1.0 for _ in range(dim)]], requires_grad=True)
        self.beta = ArjTensor([[0.0 for _ in range(dim)]], requires_grad=True)
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        output = []
        for batch in x.data:
            batch_out = []
            for token in batch:  # token: list of floats
                token_np = np.array(token)
                mean = token_np.mean()
                var = token_np.var()
                normed = (token_np - mean) / np.sqrt(var + self.eps)
                scaled = self.gamma.data[0] * normed + self.beta.data[0]
                batch_out.append(scaled.tolist())  # convert back to list
            output.append(batch_out)

        return ArjTensor(output, requires_grad=True)

    def parameters(self):
        return [self.gamma, self.beta]


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        return self.forward(x)
    def train(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True
    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None  # Useful if backprop supported later

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if self.p == 0:
            return x
        x_np = np.array(x.data)
        self.mask = (np.random.rand(*x_np.shape) > self.p).astype(np.float32)
        dropped = x_np * self.mask
        return ArjTensor(dropped.tolist(), requires_grad=True, _children=(x,), _op="dropout")



class CrossEntropyLoss:
    def __call__(self, logits, targets):
        return self.forward(logits, targets)

    def forward(self, logits, targets):
        logits_np = np.array(logits.data)     # (B, T, V)
        targets_np = np.array(targets.data).astype(np.int64)

        B, T, V = logits_np.shape

        # === 1. Softmax (numerical stability)
        max_logits = np.max(logits_np, axis=2, keepdims=True)
        exp_logits = np.exp(logits_np - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)

        # === 2. Compute loss
        flat_probs = probs.reshape(-1, V)                       # (B*T, V)
        flat_targets = targets_np.flatten()                    # (B*T,)
        correct_probs = flat_probs[np.arange(B * T), flat_targets]
        log_probs = -np.log(correct_probs + 1e-9)              # small epsilon to avoid log(0)
        avg_loss = np.mean(log_probs)

        # === 3. Gradient: dL/dlogits = probs - one_hot(targets)
        grad = probs.copy()
        grad = grad.reshape(-1, V)
        grad[np.arange(B * T), flat_targets] -= 1
        grad = grad.reshape(B, T, V) / (B * T)

        # === 4. Wrap as ArjTensor
        out = ArjTensor(avg_loss, requires_grad=True)

        def _backward():
            if logits.requires_grad:
                logits.grad = grad.tolist()

        out._backward = _backward
        out._prev = {logits}
        return out



class MSELoss:
    def __call__(self, pred, target):
        # pred, target are ArjTensors of same shape
        diff = pred - target  # ArjTensor
        sq = diff * diff  # ArjTensor (element-wise square)

        # Mean over elements (2D)
        total = sum(sum(row) for row in sq.data)  # just to compute scalar
        mean = ArjTensor(total / (len(sq.data) * len(sq.data[0])), requires_grad=True)

        # Make sure this backward links to previous tensors
        def _backward():
            grad_val = 1.0 / (len(sq.data) * len(sq.data[0]))
            sq.backward([[grad_val for _ in row] for row in sq.data])

        mean.grad = 1.0
        mean._backward = _backward
        mean._prev = {sq}
        return mean


def softmax(logits):
    """
    logits: ArjTensor of shape (batch_size, num_classes)
    Returns: ArjTensor of same shape with softmax probabilities
    """
    out = []
    for row in logits.data:
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]
        sum_exps = sum(exps)
        out.append([x / sum_exps for x in exps])
    return ArjTensor(out, requires_grad=True, _children=(logits,), _op="softmax")

class FeedForward:
    def __init__(self, d_model, hidden_dim):
        self.net = Sequential(
            Linear(d_model, hidden_dim),
            ReLU(),
            Dropout(p=0.1),
            Linear(hidden_dim, d_model)
        )

    def __call__(self, x):
        return self.net(x)

    def parameters(self):
        return self.net.parameters()
