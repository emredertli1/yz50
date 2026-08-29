

from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
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



def trace(root):
    """Grafikteki tüm düğüm ve kenarları (child -> parent) topla."""
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
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # soldan sağa
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        # her Value için bir kutu
        dot.node(name=uid, label="{ %s | data %.4f }" % (n.label, n.data), shape='record')
        if n._op:
            # işlemi temsil eden ayrı bir düğüm (ör: '+' veya '*')
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

if __name__ == "__main__":
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    e = a * b;   e.label = 'e'
    d = e + c;   d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f;   L.label = 'L'

    print(L)  # Value(data=-8.0)

    dot = draw_dot(L)
    dot.render('gorev1_graph', cleanup=True)
    print("Grafik 'gorev1_graph.svg' olarak kaydedildi.")