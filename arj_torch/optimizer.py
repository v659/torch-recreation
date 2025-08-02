import numpy as np


class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0, use_adam=False,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.initial_lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.velocities = [np.zeros_like(np.array(p.data)) for p in parameters]
        self.m = [np.zeros_like(np.array(p.data)) for p in parameters]
        self.v = [np.zeros_like(np.array(p.data)) for p in parameters]

    def step(self):
        self.t += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            # Convert to numpy arrays
            p_data = np.array(p.data)
            grad = np.array(p.grad)
            norm = np.linalg.norm(grad)
            if norm > 1.0:
                grad = grad * (1.0 / max(1.0, norm))

            if self.use_adam:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                denom = np.sqrt(v_hat) + self.eps
                update = self.lr * m_hat / denom

                p.data = (p_data - update).tolist()
                if self.weight_decay > 0:
                    p_data -= self.lr * self.weight_decay * p_data  # AdamW-style decay

            elif self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                p.data = (p_data + self.velocities[i]).tolist()

            else:
                p.data = (p_data - self.lr * grad).tolist()

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    def set_lr(self, new_lr):
        self.lr = new_lr

    def set_lr_schedule(self, step, warmup_steps=100):
        if step < warmup_steps:
            self.lr = self.initial_lr * (step / warmup_steps)
        else:
            self.lr = self.initial_lr

