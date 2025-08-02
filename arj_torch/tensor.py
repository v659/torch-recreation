import numpy as np

class ArjTensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"ArjTensor(data={self.data}, grad={self.grad})"

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def _add_grad(self, current, new):
        return current + new if current is not None else new

    def __add__(self, other):
        other = other if isinstance(other, ArjTensor) else ArjTensor(other)
        out = ArjTensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self.grad = self._add_grad(self.grad, out.grad)
            if other.requires_grad:
                other.grad = self._add_grad(other.grad, out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        other = other if isinstance(other, ArjTensor) else ArjTensor(other)
        out = ArjTensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self.grad = self._add_grad(self.grad, other.data * out.grad)
            if other.requires_grad:
                other.grad = self._add_grad(other.grad, self.data * out.grad)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, ArjTensor) else ArjTensor(other)
        out = ArjTensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op='/')

        def _backward():
            if self.requires_grad:
                self.grad = self._add_grad(self.grad, (1 / other.data) * out.grad)
            if other.requires_grad:
                other.grad = self._add_grad(other.grad, (-self.data / (other.data ** 2)) * out.grad)

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = ArjTensor(t, requires_grad=self.requires_grad, _children=(self,), _op='tanh')

        def _backward():
            if self.requires_grad:
                self.grad = self._add_grad(self.grad, (1 - t ** 2) * out.grad)

        out._backward = _backward
        return out

    def sqrt(self):
        s = np.sqrt(self.data)
        out = ArjTensor(s, requires_grad=self.requires_grad, _children=(self,), _op='sqrt')

        def _backward():
            if self.requires_grad:
                self.grad = self._add_grad(self.grad, 0.5 / s * out.grad)

        out._backward = _backward
        return out

    def sum(self):
        out = ArjTensor(np.sum(self.data), requires_grad=self.requires_grad, _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                self.grad = self._add_grad(self.grad, np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        self.grad = grad if grad is not None else np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        sliced = self.data[idx]
        return ArjTensor(sliced, requires_grad=self.requires_grad)
