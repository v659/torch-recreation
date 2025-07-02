import math


class ArjTensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = 0.0  # for scalars
        self._backward = lambda: None  # a function to compute local gradient
        self._prev = set(_children)
        self._op = _op  # operation that produced this tensor (optional debug)

    def __getitem__(self, idx):
        val = self.data[idx]
        if isinstance(val, list):
            return ArjTensor(val)
        return val


    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, ArjTensor) else ArjTensor(other)
        out = ArjTensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad,
                        _children=(self, other), _op="+")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, ArjTensor) else ArjTensor(other)
        out = ArjTensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                        _children=(self, other), _op="*")

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = ArjTensor(t, requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = 1.0

        self.grad = grad

        # Do a topological sort of the computation graph
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            t._backward()
