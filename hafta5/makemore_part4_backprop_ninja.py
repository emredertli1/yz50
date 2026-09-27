import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 1. Veri Hazırlığı
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

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
  return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# 2. Parametrelerin Tanımlanması
n_embd = 10
n_hidden = 64

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)

# Kaiming He benzeri ölçekleme
W1 = (
    torch.randn((n_embd * block_size, n_hidden), generator=g)
    * (5 / 3)
    / ((n_embd * block_size) ** 0.5)
)
b1 = torch.randn(n_hidden, generator=g) * 0.1

bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
  p.requires_grad = True

n = 32
Xb, Yb = Xtr[:n], Ytr[:n]

# 3. İleri Yayılım (Atomik Adımlar)
# Karakter indekslerini vektöre eşleyip düzleştirme
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)

# 1. Lineer katman
hprebn = embcat @ W1 + b1

# BatchNorm adımları
bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1 / (n - 1) * bndiff2.sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5) ** -0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

# Aktivasyon
h = torch.tanh(hpreact)

# 2. Lineer katman (Logits)
logits = h @ W2 + b2

# Cross-Entropy / Softmax adımları
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# PyTorch autograd referans türevleri için grafiği koruma
for t in [
    emb, embcat, hprebn, bnmeani, bndiff, bndiff2, bnvar, bnvar_inv,
    bnraw, hpreact, h, logits, logit_maxes, norm_logits, counts,
    counts_sum, counts_sum_inv, probs, logprobs,
]:
  t.retain_grad()

for p in parameters:
  p.grad = None

loss.backward()

def cmp(s, dt, t):
  exact = torch.equal(dt, t.grad)
  approx = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | EXACT: {str(exact):5s} | APPROX: {str(approx):5s} | MAX DIFF: {maxdiff:.2e}')

# 4. Manuel Geriye Yayılım (Backpropagation by Hand)

# dLoss / dlogprobs
# loss = - (1/n) * sum(logprobs[i, Yb[i]])
# Sadece doğru sınıfın indeksi loss'a girer, türevi -1/n, diğerleri 0'dır.
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n
cmp('logprobs', dlogprobs, logprobs)

# dLoss / dprobs
# logprobs = log(probs) -> d/dx [ln(x)] = 1/x
dprobs = (1.0 / probs) * dlogprobs
cmp('probs', dprobs, probs)

# dLoss / dcounts_sum_inv
# probs = counts * counts_sum_inv
# counts_sum_inv (32, 1) her satırdaki 27 elemana broadcast edildiğinden dim=1 boyunca toplanır.
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

# dLoss / dcounts_sum
# counts_sum_inv = (counts_sum)**(-1) -> türevi -1 / (counts_sum^2)
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)

# dLoss / dcounts
# counts iki yere dallanır: (1) probs çarpımı, (2) counts_sum satır toplamı.
# Çok değişkenli zincir kuralı gereği iki koldan gelen gradyanlar toplanır.
dcounts = counts_sum_inv * dprobs + torch.ones_like(counts) * dcounts_sum
cmp('counts', dcounts, counts)

# dLoss / dnorm_logits
# counts = exp(norm_logits) -> d/dx [e^x] = e^x
dnorm_logits = counts * dcounts
cmp('norm_logits', dnorm_logits, norm_logits)

# dLoss / dlogit_maxes
# norm_logits = logits - logit_maxes
# logit_maxes (32, 1) satırdaki 27 sütundan çıkarıldığı için türevi -1'dir ve dim=1 boyunca toplanır.
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
cmp('logit_maxes', dlogit_maxes, logit_maxes)

# dLoss / dlogits
# logits iki yoldan gradyan alır:
# 1) norm_logits = logits - logit_maxes (türev 1)
# 2) logit_maxes = max(logits) -> türev sadece maksimum elemanın indeksinde 1, diğerlerinde 0'dır (one-hot maske).
dlogits = dnorm_logits.clone()
dlogits += F.one_hot(logits.max(1).indices, num_classes=vocab_size) * dlogit_maxes
cmp('logits', dlogits, logits)

# 2. Lineer katman türevleri
# logits = h @ W2 + b2
# Boyutlar: logits (32, 27), h (32, 64), W2 (64, 27), b2 (27)
dh = dlogits @ W2.T
cmp('h', dh, h)

dW2 = h.T @ dlogits
cmp('W2', dW2, W2)

# b2 batch boyunca kopyalandığı için batch ekseninde (dim=0) toplanır.
db2 = dlogits.sum(0)
cmp('b2', db2, b2)

# Tanh aktivasyon türevi
# h = tanh(hpreact) -> d/dx [tanh(x)] = 1 - tanh^2(x) = 1 - h^2
dhpreact = (1.0 - h**2) * dh
cmp('hpreact', dhpreact, hpreact)

# BatchNorm ölçek ve kaydırma türevleri
# hpreact = bngain * bnraw + bnbias
# bngain ve bnbias (1, 64) batch boyunca broadcast edildiği için dim=0 boyunca toplanır.
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
cmp('bngain', dbngain, bngain)

dbnbias = dhpreact.sum(0, keepdim=True)
cmp('bnbias', dbnbias, bnbias)

dbnraw = bngain * dhpreact
cmp('bnraw', dbnraw, bnraw)

# dLoss / dbnvar_inv
# bnraw = bndiff * bnvar_inv
# bnvar_inv (1, 64) batch'e yayıldığı için dim=0 boyunca toplanır.
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

# dLoss / dbnvar
# bnvar_inv = (bnvar + eps)**(-0.5) -> türevi -0.5 * (bnvar + eps)^(-1.5)
dbnvar = (-0.5 * (bnvar + 1e-5) ** -1.5) * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)

# dLoss / dbndiff2
# bnvar = (1 / (n - 1)) * sum(bndiff2)
dbndiff2 = (1.0 / (n - 1)) * torch.ones_like(bndiff2) * dbnvar
cmp('bndiff2', dbndiff2, bndiff2)

# dLoss / dbndiff
# bndiff iki kola ayrılır: (1) bndiff2 = bndiff^2, (2) bnraw = bndiff * bnvar_inv
dbndiff = (2.0 * bndiff) * dbndiff2 + bnvar_inv * dbnraw
cmp('bndiff', dbndiff, bndiff)

# dLoss / dbnmeani
# bndiff = hprebn - bnmeani
# bnmeani (1, 64) her örnekten çıkarıldığından türevi -1'dir ve batch boyunca toplanır.
dbnmeani = (-dbndiff).sum(0, keepdim=True)
cmp('bnmeani', dbnmeani, bnmeani)

# dLoss / dhprebn
# hprebn iki koldan gelir: (1) bndiff farkı (türev 1), (2) bnmeani ortalaması (türev 1/n)
dhprebn = dbndiff.clone() + (1.0 / n) * torch.ones_like(hprebn) * dbnmeani
cmp('hprebn', dhprebn, hprebn)

# 1. Lineer katman türevleri
# hprebn = embcat @ W1 + b1
# Boyutlar: hprebn (32, 64), embcat (32, 30), W1 (30, 64), b1 (64)
dW1 = embcat.T @ dhprebn
cmp('W1', dW1, W1)

db1 = dhprebn.sum(0)
cmp('b1', db1, b1)

dembcat = dhprebn @ W1.T
cmp('embcat', dembcat, embcat)

# dLoss / demb
# embcat = emb.view(emb.shape[0], -1)
# View işlemi yalnızca boyut indekslerini değiştirdiğinden gradyan da doğrudan eski şekle çevrilir.
demb = dembcat.view(emb.shape)
cmp('emb', demb, emb)

# dLoss / dC (Embedding matrisi)
# emb = C[Xb]
# Aynı karakter mini-batch içinde birden fazla kez geçtiğinde gradyanları toplanır (accumulate).
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
  for j in range(Xb.shape[1]):
    ix = Xb[k, j]
    dC[ix] += demb[k, j]
cmp('C', dC, C)

print('\nTüm analitik türevler PyTorch referansı ile doğrulandı.')