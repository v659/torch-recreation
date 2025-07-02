import random
import math
from arj_torch.tensor import ArjTensor


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = ArjTensor([[random.uniform(-1, 1) for _ in range(in_features)] for _ in range(out_features)])
        self.bias = ArjTensor([[0.0 for _ in range(out_features)]])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        output = []
        for row in x.data:
            out_row = []
            for j in range(self.out_features):
                val = sum(row[i] * self.weight[j][i] for i in range(self.in_features))
                val += self.bias[0][j]
                out_row.append(val)
            output.append(out_row)
        return ArjTensor(output)

    def parameters(self):
        return [self.weight, self.bias]


class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = ArjTensor([[random.uniform(-1, 1) for _ in range(embedding_dim)] for _ in range(vocab_size)])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        output = []
        for row in x.data:
            out_row = []
            for token in row:
                out_row.extend(self.weight.data[token])
            output.append(out_row)
        return ArjTensor(output)

    def parameters(self):
        return [self.weight]


class ReLU:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return ArjTensor([[max(0, v) for v in row] for row in x.data])


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = ArjTensor([[1.0] * dim])
        self.beta = ArjTensor([[0.0] * dim])

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        output = []
        for row in x.data:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            normed = [(v - mean) / ((var + self.eps) ** 0.5) for v in row]
            normed = [self.gamma.data[0][i] * normed[i] + self.beta.data[0][i] for i in range(len(row))]
            output.append(normed)
        return ArjTensor(output)

    def parameters(self):
        return [self.gamma, self.beta]


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
        return ArjTensor(data)


class CrossEntropyLoss:
    def __call__(self, logits, targets):
        return self.forward(logits, targets)

    def forward(self, logits, targets):
        losses = []
        for i in range(len(logits.data)):
            logit_row = logits.data[i]
            target_idx = targets.data[i][0] if isinstance(targets.data[i], list) else targets.data[i]
            max_logit = max(logit_row)
            exps = [math.exp(l - max_logit) for l in logit_row]
            log_sum_exp = math.log(sum(exps))
            loss = - (logit_row[target_idx] - max_logit - log_sum_exp)
            losses.append(loss)
        return sum(losses) / len(losses)


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if hasattr(param, 'grad'):
                for i in range(len(param.data)):
                    for j in range(len(param.data[i])):
                        param.data[i][j] -= self.lr * param.grad[i][j]

    def zero_grad(self):
        for param in self.parameters:
            if hasattr(param, 'grad'):
                for i in range(len(param.data)):
                    for j in range(len(param.data[i])):
                        param.grad[i][j] = 0.0


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


class MSELoss:
    def __call__(self, pred, target):
        diff = [(p - t)**2 for pr, tr in zip(pred.data, target.data) for p, t in zip(pr, tr)]
        return sum(diff) / len(diff)


