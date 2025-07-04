import random
import math
from arj_torch.tensor import ArjTensor


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = ArjTensor([[random.uniform(-1, 1) for _ in range(in_features)] for _ in range(out_features)], requires_grad=True)
        self.bias = ArjTensor([[0.0 for _ in range(out_features)]], requires_grad=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.input = x  # Save for backward
        input_data = x.data

        # Check if input is 3D (B, T, C)
        is_3d = isinstance(input_data[0][0], list)

        if is_3d:
            B, T, C = len(input_data), len(input_data[0]), len(input_data[0][0])
            output = []
            for b in range(B):
                time_out = []
                for t in range(T):
                    row = input_data[b][t]  # row is of length C
                    assert len(row) == self.in_features, f"Linear expected input dim={self.in_features}, got {len(row)}"
                    out_row = []
                    for j in range(self.out_features):
                        val = sum(row[i] * self.weight.data[j][i] for i in range(self.in_features)) + self.bias.data[0][
                            j]
                        out_row.append(val)
                    time_out.append(out_row)
                output.append(time_out)
        else:
            B, C = len(input_data), len(input_data[0])
            output = []
            for b in range(B):
                row = input_data[b]
                assert len(row) == self.in_features, f"Linear expected input dim={self.in_features}, got {len(row)}"
                out_row = []
                for j in range(self.out_features):
                    val = sum(row[i] * self.weight.data[j][i] for i in range(self.in_features)) + self.bias.data[0][j]
                    out_row.append(val)
                output.append(out_row)

        out = ArjTensor(output, requires_grad=True, _children=(x, self.weight, self.bias), _op="linear")

        def _backward():
            if self.weight.grad is None:
                self.weight.grad = [[0.0 for _ in range(self.in_features)] for _ in range(self.out_features)]
            if self.bias.grad is None:
                self.bias.grad = [[0.0 for _ in range(self.out_features)]]
            if not x.requires_grad:
                return

            # Initialize grads
            if is_3d:
                B, T, _ = len(output), len(output[0]), len(output[0][0])
            else:
                B, T = len(output), 1  # treat 2D as (B, 1)

            # Gradient for weight and bias
            for j in range(self.out_features):
                for i in range(self.in_features):
                    grad_sum = 0.0
                    for b in range(B):
                        time_steps = T if is_3d else 1
                        for t in range(time_steps):
                            inp = self.input.data[b][t][i] if is_3d else self.input.data[b][i]
                            grad = out.grad[b][t][j] if is_3d else out.grad[b][j]
                            grad_sum += inp * grad
                    self.weight.grad[j][i] += grad_sum

            for j in range(self.out_features):
                grad_sum = 0.0
                for b in range(B):
                    for t in range(T):
                        grad_sum += out.grad[b][t][j] if is_3d else out.grad[b][j]
                self.bias.grad[0][j] += grad_sum

            # Gradient for input x
            grad_input = []
            for b in range(B):
                time_steps = T if is_3d else 1
                time_out = []
                for t in range(time_steps):
                    grad_row = []
                    for i in range(self.in_features):
                        grad = 0.0
                        for j in range(self.out_features):
                            g = out.grad[b][t][j] if is_3d else out.grad[b][j]
                            grad += g * self.weight.data[j][i]
                        grad_row.append(grad)
                    time_out.append(grad_row)
                grad_input.append(time_out if is_3d else time_out[0])

            x.grad = grad_input

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
        output = []
        for row in x.data:
            out_row = []
            for token in row:
                out_row.extend(self.weight.data[token])
            output.append(out_row)

        out = ArjTensor(output, requires_grad=True, _children=(self.weight,), _op="embedding")

        def _backward():
            # (You can skip backward here for now if you're not fine-tuning embeddings)
            pass

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]


class ReLU:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out_data = []

        if isinstance(x.data[0][0], list):  # 3D (B, T, C)
            for row in x.data:
                new_row = []
                for token in row:
                    new_token = [max(0.0, v) for v in token]
                    new_row.append(new_token)
                out_data.append(new_row)
        else:  # 2D (B, C)
            for row in x.data:
                out_data.append([max(0.0, v) for v in row])

        return ArjTensor(out_data, requires_grad=True, _children=(x,), _op="relu")




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

        # Support (B, T, C)
        for batch in x.data:
            batch_out = []
            for token in batch:  # token: list of floats (C,)
                mean = sum(token) / len(token)
                var = sum((v - mean) ** 2 for v in token) / len(token)
                normed = [(v - mean) / ((var + self.eps) ** 0.5) for v in token]
                normed = [
                    self.gamma.data[0][i] * normed[i] + self.beta.data[0][i]
                    for i in range(self.dim)
                ]


                batch_out.append(normed)
            output.append(batch_out)

        return ArjTensor(output, requires_grad=True)

    def parameters(self):
        return [self.gamma, self.beta]


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        return self.forward(x)

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

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if self.p == 0:
            return x
        data = []
        for row in x.data:
            data.append([v * (random.random() > self.p) for v in row])
        return ArjTensor(data, requires_grad=True)


class CrossEntropyLoss:
    def __call__(self, logits, targets):
        return self.forward(logits, targets)

    def forward(self, logits, targets):
        batch_size = len(logits.data)
        loss_sum = 0.0
        grads = [[0.0 for _ in row] for row in logits.data]

        for i in range(batch_size):
            row = logits.data[i]
            target = targets.data[i][0] if isinstance(targets.data[i], list) else targets.data[i]
            max_logit = max(row)
            exps = [math.exp(x - max_logit) for x in row]
            sum_exps = sum(exps)
            probs = [e / sum_exps for e in exps]
            log_prob = math.log(probs[target])
            loss_sum -= log_prob

            # compute softmax gradient
            for j in range(len(row)):
                grads[i][j] = probs[j]
            grads[i][target] -= 1  # ∂L/∂logit = softmax - target

        avg_loss = loss_sum / batch_size
        out = ArjTensor(avg_loss, requires_grad=True)

        def _backward():
            if logits.requires_grad:
                scale = 1.0 / batch_size
                logits.grad = [[g * scale for g in row] for row in grads]

        out._backward = _backward
        out._prev = {logits}
        return out


class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0, use_adam=False, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # timestep

        self.velocities = [ArjTensor._zeros_like(p, p.data) for p in parameters]
        self.m = [ArjTensor._zeros_like(p, p.data) for p in parameters]  # Adam: first moment
        self.v = [ArjTensor._zeros_like(p, p.data) for p in parameters]  # Adam: second moment

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad

            # 🔒 Force ArjTensor type
            if not isinstance(grad, ArjTensor):
                grad = ArjTensor(grad, requires_grad=False)
            if not isinstance(self.m[i], ArjTensor):
                self.m[i] = ArjTensor(self.m[i], requires_grad=False)
            if not isinstance(self.v[i], ArjTensor):
                self.v[i] = ArjTensor(self.v[i], requires_grad=False)

            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + (self.weight_decay * p)

            if self.use_adam:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                update = self.lr * m_hat / (v_hat.sqrt() + self.eps)
                p.data = [[pij - uij for pij, uij in zip(p_row, u_row)]
                          for p_row, u_row in zip(p.data, update.data)]

            elif self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                p.data = [[pij + vij for pij, vij in zip(p_row, v_row)]
                          for p_row, v_row in zip(p.data, self.velocities[i].data)]

            else:
                p.data = [[pij - self.lr * gij for pij, gij in zip(p_row, g_row)]
                          for p_row, g_row in zip(p.data, grad.data)]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None






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

