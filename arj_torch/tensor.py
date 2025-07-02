import math

class ArjTensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = data  # scalar only for now
        self.requires_grad = requires_grad
        self.grad = 0.0  # scalar gradient only
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, ArjTensor) else ArjTensor(other)

        out = ArjTensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, other):
        other = other if isinstance(other, ArjTensor) else ArjTensor(other)

        out = ArjTensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = ArjTensor(t, requires_grad=self.requires_grad, _children=(self,), _op='tanh')

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        # Initialize gradient
        self.grad = 1.0 if grad is None else grad

        # Topological order
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        # Backpropagate
        for t in reversed(topo):
            t._backward()
