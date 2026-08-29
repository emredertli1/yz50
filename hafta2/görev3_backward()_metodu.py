
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
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()

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


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

print("=" * 60)
print("TEST 1: L = (a*b + c) * f")
print("=" * 60)

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b;   e.label = 'e'
d = e + c;   d.label = 'd'
f = Value(-2.0, label='f')
L = d * f;   L.label = 'L'

L.backward()

print(f"dL/da = {a.grad}  (beklenen: 6.0)")
print(f"dL/db = {b.grad}  (beklenen: -4.0)")
print(f"dL/dc = {c.grad}  (beklenen: -2.0)")
print(f"dL/df = {f.grad}  (beklenen: 4.0)")
assert a.grad == 6.0 and b.grad == -4.0 and c.grad == -2.0 and f.grad == 4.0
print("✅ Doğru.\n")
print("=" * 60)
print("TEST 2: Nöron, o = tanh(x1*w1 + x2*w2 + b)")
print("=" * 60)

x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
bias = Value(6.8813735870195432, label='b')

x1w1 = x1 * w1;         x1w1.label = 'x1*w1'
x2w2 = x2 * w2;         x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1+x2*w2'
n = x1w1x2w2 + bias;    n.label = 'n'
o = n.tanh();           o.label = 'o'

o.backward()

print(f"o = {round(o.data, 4)}")
print(f"do/dx1 = {round(x1.grad, 4)}  (beklenen: -1.5)")
print(f"do/dw1 = {round(w1.grad, 4)}  (beklenen: 1.0)")
print(f"do/dx2 = {round(x2.grad, 4)}  (beklenen: 0.5)")
print(f"do/dw2 = {round(w2.grad, 4)}  (beklenen: 0.0)")
assert abs(x1.grad - (-1.5)) < 1e-6 and abs(w2.grad - 0.0) < 1e-6
print("✅ Doğru.\n")

dot = draw_dot(o)
dot.render('gorev3_neuron_backward', cleanup=True)
print("Grafik 'gorev3_neuron_backward.svg' olarak kaydedildi.\n")

print("=" * 60)
print("TEST 3: b = a + a  =>  b = 2a  =>  db/da 2 olmalı")
print("=" * 60)

a2 = Value(3.0, label='a')
b2 = a2 + a2  
b2.backward()

print(f"a.grad = {a2.grad}  (beklenen: 2.0)")
assert a2.grad == 2.0
print("✅ Doğru: += sayesinde iki katkı (self ve other üzerinden gelen)")
print("   üzerine yazılmadı, birbirine EKLENDİ.")