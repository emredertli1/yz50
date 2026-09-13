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

# --- Görev 3: train / dev / test bölünmesi ---
import random
random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])       # %80 train
Xdev, Ydev = build_dataset(words[n1:n2])   # %10 dev
Xte, Yte = build_dataset(words[n2:])       # %10 test

# --- Görev 2: parametreler ---
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

# --- Adım A: tek minibatch'i overfit et (sanity check) ---
print("=== Overfit testi (32 örnek) ===")
Xb, Yb = Xtr[:32], Ytr[:32]

for i in range(1000):
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    for p in parameters:
        p.data += -0.1 * p.grad

print("overfit loss:", loss.item())

# --- Adım B: parametreleri sıfırdan tekrar başlat ---
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# --- Adım C: learning rate taraması ---
print("\n=== Learning rate taraması ===")
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lossi = []
lrei = []

for i in range(1000):
    ix = torch.randint(0, Xtr.shape[0], (32,))

    emb = C[Xtr[ix]]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = lrs[i]
    for p in parameters:
        p.data += -lr * p.grad

    lrei.append(lre[i].item())
    lossi.append(loss.item())

print("tarama bitti, en iyi lr'yi grafikten seç (yaklaşık 0.1 civarı bekleniyor)")

# --- Adım D: parametreleri tekrar sıfırla, gerçek eğitime başla ---
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

print("\n=== Gerçek eğitim (tüm train verisiyle, minibatch) ===")
lr = 0.1
for i in range(30000):
    ix = torch.randint(0, Xtr.shape[0], (32,))

    emb = C[Xtr[ix]]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    if i > 20000:
        lr = 0.01  # eğitim ilerledikçe lr'yi düşür

    for p in parameters:
        p.data += -lr * p.grad

    if i % 5000 == 0:
        print(f"iter {i}: train loss {loss.item():.4f}")

# --- Adım E: dev üzerinde loss raporla ---
emb = C[Xdev]
embcat = emb.view(emb.shape[0], -1)
h = torch.tanh(embcat @ W1 + b1)
logits = h @ W2 + b2
dev_loss = F.cross_entropy(logits, Ydev)

print("\n=== Sonuç ===")
print("dev loss:", dev_loss.item())