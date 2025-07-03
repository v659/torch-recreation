import math


class ArjTensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = self._zeros_like(data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def _zeros_like(self, data):
        if isinstance(data, float):
            return 0.0
        elif isinstance(data, list):
            return [[0.0 for _ in row] for row in data]
        else:
            raise TypeError("Unsupported data type for grad")

    def _add_grads(self, a, b):
        if isinstance(a, float):
            return a + b
        elif isinstance(a, list):
            return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
        else:
            raise TypeError("Unsupported grad type")

    def __getitem__(self, idx):
        val = self.data[idx]
        if isinstance(val, list):
            return ArjTensor(val)
        return val

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, ArjTensor) else ArjTensor(other)

        if isinstance(self.data, float) and isinstance(other.data, float):
            out_data = self.data + other.data
        elif isinstance(self.data, list) and isinstance(other.data, float):
            out_data = [[val + other.data for val in row] for row in self.data]
        elif isinstance(self.data, float) and isinstance(other.data, list):
            out_data = [[self.data + val for val in row] for row in other.data]
        elif isinstance(self.data, list) and isinstance(other.data, list):
            out_data = [
                [self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))]
                for i in range(len(self.data))
            ]
        else:
            raise TypeError("Unsupported types in __add__")

        out = ArjTensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                        _children=(self, other), _op="+")

        def _backward():
            if self.requires_grad:
                self.grad = self._add_grads(self.grad, out.grad)
            if other.requires_grad:
                other.grad = self._add_grads(other.grad, out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, other):
        if not isinstance(other, ArjTensor):
            other = ArjTensor(float(other))  # convert int to float inside a tensor

        if isinstance(self.data, float) and isinstance(other.data, float):
            out_data = self.data * other.data
        elif isinstance(self.data, list) and isinstance(other.data, float):
            out_data = [[val * other.data for val in row] for row in self.data]
        elif isinstance(self.data, float) and isinstance(other.data, list):
            out_data = [[self.data * val for val in row] for row in other.data]
        elif isinstance(self.data, list) and isinstance(other.data, list):
            out_data = [
                [self.data[i][j] * other.data[i][j] for j in range(len(self.data[0]))]
                for i in range(len(self.data))
            ]
        else:
            raise TypeError(f"Unsupported types in __mul__: {type(self.data)}, {type(other.data)}, {other.data}")

        out = ArjTensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                        _children=(self, other), _op="*")

        def _backward():
            if self.requires_grad and isinstance(out.grad, list):
                if isinstance(other.data, float):
                    grad = [[other.data * out.grad[i][j] for j in range(len(out.grad[0]))] for i in
                            range(len(out.grad))]
                else:
                    grad = [[other.data[i][j] * out.grad[i][j] for j in range(len(out.grad[0]))] for i in
                            range(len(out.grad))]
                self.grad = self._add_grads(self.grad, grad)

            if other.requires_grad and isinstance(out.grad, list):
                if isinstance(self.data, float):
                    grad = [[self.data * out.grad[i][j] for j in range(len(out.grad[0]))] for i in range(len(out.grad))]
                else:
                    grad = [[self.data[i][j] * out.grad[i][j] for j in range(len(out.grad[0]))] for i in
                            range(len(out.grad))]
                other.grad = self._add_grads(other.grad, grad)

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

    def sum(self):
        if isinstance(self.data, float):
            return self  # already scalar

        total = sum(sum(row) for row in self.data)
        out = ArjTensor(total, requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                grad = [[1.0 for _ in row] for row in self.data]
                self.grad = self._add_grads(self.grad, grad)

        out._backward = _backward
        return out


    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = 1.0 if isinstance(self.data, float) else self._zeros_like(self.data)

        self.grad = grad

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
