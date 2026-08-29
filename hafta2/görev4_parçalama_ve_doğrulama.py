

import math
from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other ** -1

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def tanh_parcali(x):
    """tanh(x) = (e^2x - 1) / (e^2x + 1), sadece exp/-/+//,** kullanarak."""
    e2x = (x * 2).exp()
    return (e2x - 1) / (e2x + 1)


print("=" * 60)
print("Tek parça tanh() vs exp/div/pow ile parçalanmış tanh")
print("=" * 60)

x1 = Value(2.0, label='x1'); w1 = Value(-3.0, label='w1')
x2 = Value(0.0, label='x2'); w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')
n = x1 * w1 + x2 * w2 + b
o1 = n.tanh()
o1.backward()
grad_tek = (x1.grad, w1.grad, x2.grad, w2.grad)

x1 = Value(2.0); w1 = Value(-3.0)
x2 = Value(0.0); w2 = Value(1.0)
b = Value(6.8813735870195432)
n = x1 * w1 + x2 * w2 + b
o2 = tanh_parcali(n)
o2.backward()
grad_parcali = (x1.grad, w1.grad, x2.grad, w2.grad)

print(f"o (tek parça)     = {round(o1.data, 6)}")
print(f"o (parçalanmış)   = {round(o2.data, 6)}")
print(f"grad (tek parça)  = {tuple(round(g,6) for g in grad_tek)}")
print(f"grad (parçalanmış)= {tuple(round(g,6) for g in grad_parcali)}")
assert all(abs(a - b) < 1e-9 for a, b in zip(grad_tek, grad_parcali))
print("✅ İkisi birebir aynı -- tanh'ı exp/div/pow'a bölmek sonucu değiştirmedi.\n")



print("=" * 60)
print("DOĞRULAMA: backward() vs numerical derivative")
print("=" * 60)

vals = dict(x1=2.0, w1=-3.0, x2=0.0, w2=1.0, b=6.8813735870195432)


x1 = Value(vals['x1']); w1 = Value(vals['w1'])
x2 = Value(vals['x2']); w2 = Value(vals['w2'])
b = Value(vals['b'])
o = tanh_parcali(x1 * w1 + x2 * w2 + b)
o.backward()
grad_backward = dict(x1=x1.grad, w1=w1.grad, x2=x2.grad, w2=w2.grad, b=b.grad)


def f(x1, w1, x2, w2, b):
    x1 = Value(x1); w1 = Value(w1); x2 = Value(x2); w2 = Value(w2); b = Value(b)
    return tanh_parcali(x1 * w1 + x2 * w2 + b).data

def numerical_grad(var_name, h=1e-6):
    v = dict(vals)
    v[var_name] += h; f1 = f(**v)
    v[var_name] -= 2*h; f2 = f(**v)
    return (f1 - f2) / (2*h)

grad_numerical = {k: numerical_grad(k) for k in vals}

print(f"{'değişken':<8}{'backward()':>14}{'numerical':>14}")
for k in vals:
    print(f"{k:<8}{grad_backward[k]:>14.6f}{grad_numerical[k]:>14.6f}")

for k in vals:
    assert abs(grad_backward[k] - grad_numerical[k]) < 1e-4
print("\n✅ İkisi eşleşiyor.")
