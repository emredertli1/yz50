import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

#  VERİ HAZIRLIĞI (block_size = 8 yapıldı)

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Bağlam 3 harften 8 harfe çıkarıldı
block_size = 8

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

print(f'Eğitim kümesi boyutu: {Xtr.shape}, {Ytr.shape}')


# KATMAN SINIFLARI (PyTorchifikasyon)

class Linear:

  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out)) / (fan_in**0.5)
    self.bias = torch.zeros(fan_out) if bias else None

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  def __call__(self, x):
    if self.training:
      xmean = x.mean(0, keepdim=True)
      xvar = x.var(0, keepdim=True, unbiased=False)
    else:
      xmean = self.running_mean
      xvar = self.running_var

    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta

    if self.training:
      with torch.no_grad():
        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * xmean
        self.running_var = (
            1 - self.momentum
        ) * self.running_var + self.momentum * xvar

    return self.out

  def parameters(self):
    return [self.gamma, self.beta]


class Tanh:

  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out

  def parameters(self):
    return []


class Embedding:

  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))

  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out

  def parameters(self):
    return [self.weight]


class Flatten:

  def __call__(self, x):
    self.out = x.view(x.shape[0], -1)
    return self.out

  def parameters(self):
    return []


class Sequential:

  def __init__(self, layers):
    self.layers = layers

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]


# MODEL TANIMI

torch.manual_seed(42)

n_embd = 10
n_hidden = 200
vocab_size = 27

# İlk katman 8 * 10 = 80 giriş alır
model = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * block_size, n_hidden, bias=False),  # 80 -> 200
    BatchNorm1d(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size, bias=True),  # 200 -> 27
])

# Son katmanın aşırı özgüvenli başlamasını önleme
with torch.no_grad():
  model.layers[-1].weight *= 0.01

parameters = model.parameters()
print(f'Toplam parametre sayısı: {sum(p.numel() for p in parameters)}')

for p in parameters:
  p.requires_grad = True

#  EĞİTİM DÖNGÜSÜ

max_steps = 30000
batch_size = 32
lossi = []

print('\nEğitim başladı...')
for i in range(max_steps):
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))
  Xb, Yb = Xtr[ix], Ytr[ix]

  # İleri yayılım (Agnostik)
  logits = model(Xb)
  loss = F.cross_entropy(logits, Yb)

  # Geriye yayılım
  for p in parameters:
    p.grad = None
  loss.backward()

  # Öğrenme oranı takvimi
  lr = 0.1 if i < 20000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  lossi.append(loss.item())

#  DEĞERLENDİRME (EVAL)

for layer in model.layers:
  if hasattr(layer, 'training'):
    layer.training = False


@torch.no_grad()
def evaluate_split(split):
  x, y = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev), 'test': (Xte, Yte)}[split]
  logits = model(x)
  loss = F.cross_entropy(logits, y)
  print(f'{split.upper()} loss: {loss.item():.4f}')


print('\n=== SONUÇLAR ===')
evaluate_split('train')
evaluate_split('val')

# ============================================================
# 6. DÜZELTİLMİŞ LOSS EĞRİSİ ÇİZİMİ
# ============================================================
plt.figure(figsize=(10, 4))
plt.plot(torch.tensor(lossi).view(-1, 200).mean(1))
plt.title('Düzeltilmiş Loss Eğrisi (200 Adımlık Ortalamalar - Baseline)')
plt.xlabel('İterasyon / 200')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 7. MODELİ KONUŞTURMA (SAMPLING / İSİM ÜRETME)
# ============================================================
g = torch.Generator().manual_seed(2147483647 + 10)

print('\n=== ÜRETİLEN YENİ İSİMLER ===')
for _ in range(20):
  out = []
  context = [0] * block_size  # 8 elemanlı boş bağlam: '........'

  while True:
    # Modele tek bir örnek verileceği için şekil: (1, 8)
    x = torch.tensor([context])
    logits = model(x)
    probs = F.softmax(logits, dim=1)

    # Olasılık dağılımından zar at
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()

    context = context[1:] + [ix]
    out.append(ix)

    if ix == 0:  # '.' geldiğinde isim sonlanır
      break

  print(''.join(itos[i] for i in out[:-1]))