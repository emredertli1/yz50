import torch
import torch.nn.functional as F

# --- Görev 1: veri seti + embedding ---
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

X, Y = build_dataset(words)

# --- Görev 2: gizli katman + çıkış katmanı ---
n_embd = 2
n_hidden = 100

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# ileri geçiş
emb = C[X]
embcat = emb.view(emb.shape[0], -1)
hpreact = embcat @ W1 + b1
h = torch.tanh(hpreact)
logits = h @ W2 + b2

# loss - elle
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
loss_manual = -probs[torch.arange(logits.shape[0]), Y].log().mean()

# loss - F.cross_entropy ile
loss_builtin = F.cross_entropy(logits, Y)

print(loss_manual.item())
print(loss_builtin.item())