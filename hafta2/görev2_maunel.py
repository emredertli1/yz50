

import math
from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0        
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')
        return out



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
print("ÖRNEK 1: L = (a*b + c) * f  -- gradyanlar ELLE dolduruluyor")
print("=" * 60)

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a * b;   e.label = 'e'      
d = e + c;   d.label = 'd'      
f = Value(-2.0, label='f')
L = d * f;   L.label = 'L'     

L.grad = 1.0
d.grad = f.data          
f.grad = d.data         
e.grad = d.grad * 1.0   
c.grad = d.grad * 1.0    

a.grad = e.grad * b.data  
b.grad = e.grad * a.data   

print(f"L = {L.data}")
print(f"dL/da = {a.grad}  (beklenen: 6.0)")
print(f"dL/db = {b.grad}  (beklenen: -4.0)")
print(f"dL/dc = {c.grad}  (beklenen: -2.0)")
print(f"dL/dd = {d.grad}  (beklenen: -2.0)")
print(f"dL/de = {e.grad}  (beklenen: -2.0)")
print(f"dL/df = {f.grad}  (beklenen: 4.0)")

dot1 = draw_dot(L)
dot1.render('gorev2_basit_ifade', cleanup=True)
print("\nGrafik 'gorev2_basit_ifade.svg' olarak kaydedildi.\n")
print("=" * 60)
print("ÖRNEK 2: Tek nöron, o = tanh(x1*w1 + x2*w2 + b) -- elle chain rule")
print("=" * 60)

x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')

x1w1 = x1 * w1;         x1w1.label = 'x1*w1'
x2w2 = x2 * w2;         x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1+x2*w2'
n = x1w1x2w2 + b;       n.label = 'n'
o = n.tanh();           o.label = 'o'
o.grad = 1.0

n.grad = (1 - o.data ** 2) * o.grad

x1w1x2w2.grad = n.grad * 1.0
b.grad = n.grad * 1.0

x1w1.grad = x1w1x2w2.grad * 1.0
x2w2.grad = x1w1x2w2.grad * 1.0
x1.grad = x1w1.grad * w1.data
w1.grad = x1w1.grad * x1.data
x2.grad = x2w2.grad * w2.data
w2.grad = x2w2.grad * x2.data

print(f"o = {round(o.data, 4)}")
print(f"do/dx1 = {round(x1.grad, 4)}  (beklenen: -1.5)")
print(f"do/dw1 = {round(w1.grad, 4)}  (beklenen: 1.0)")
print(f"do/dx2 = {round(x2.grad, 4)}  (beklenen: 0.5)")
print(f"do/dw2 = {round(w2.grad, 4)}  (beklenen: 0.0)")

dot2 = draw_dot(o)
dot2.render('gorev2_neuron', cleanup=True)
print("\nGrafik 'gorev2_neuron.svg' olarak kaydedildi.")