import random
import torch
import torch.nn.functional as F

# 1. TÜRKÇE VERİ SETİ VE BAĞLAM (block_size = 8)

words = open('turkish_names.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.strip().lower() for w in words if w.strip()]

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 8  # WaveNet hiyerarşisi için 8 karakterlik pencere

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

print(f'Türkçe İsim Sayısı: {len(words)}')
print(f'Alfabe Boyutu     : {vocab_size}')
print(f'Bağlam Genişliği  : {block_size}\n')

# 2. WAVENET BİLEŞENLERİ (KATMAN SINIFLARI)

n_embd = 24
n_hidden = 128


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


class FlattenConsecutive:

  def __init__(self, n):
    self.n = n

  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T // self.n, C * self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
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


class BatchNorm1dCorrect:

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
      dim_reduce = (0, 1) if x.ndim == 3 else 0
      xmean = x.mean(dim_reduce, keepdim=True)
      xvar = x.var(dim_reduce, keepdim=True, unbiased=False)
    else:
      xmean = self.running_mean
      xvar = self.running_var

    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta

    if self.training:
      with torch.no_grad():
        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * xmean.squeeze()
        self.running_var = (
            1 - self.momentum
        ) * self.running_var + self.momentum * xvar.squeeze()

    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

# 3. TÜRKÇE WAVENET MODELİNİN KURULMASI

torch.manual_seed(42)

model = Sequential([
    Embedding(vocab_size, n_embd),  # (B, 8, 24)
    FlattenConsecutive(2),  # (B, 4, 48)
    Linear(n_embd * 2, n_hidden, bias=False),  # (B, 4, 128)
    BatchNorm1dCorrect(n_hidden),
    Tanh(),
    FlattenConsecutive(2),  # (B, 2, 256)
    Linear(n_hidden * 2, n_hidden, bias=False),  # (B, 2, 128)
    BatchNorm1dCorrect(n_hidden),
    Tanh(),
    FlattenConsecutive(2),  # (B, 128)
    Linear(n_hidden * 2, n_hidden, bias=False),  # (B, 128)
    BatchNorm1dCorrect(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),  # (B, vocab_size)
])

with torch.no_grad():
  model.layers[-1].weight *= 0.1  # Başlangıç loss kararlılığı

parameters = model.parameters()
for p in parameters:
  p.requires_grad = True

print(f'WaveNet Toplam Parametre Sayısı: {sum(p.nelement() for p in parameters):,}')


# 4. EĞİTİM DÖNGÜSÜ (30.000 Adım)

g = torch.Generator().manual_seed(42)
steps = 30000
batch_size = 32

print('\nWaveNet Türkçe veri kümesiyle eğitiliyor...')
for i in range(steps):
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix]

  logits = model(Xb)
  loss = F.cross_entropy(logits, Yb)

  for p in parameters:
    p.grad = None
  loss.backward()

  lr = 0.1 if i < 20000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  if i % 5000 == 0:
    print(f'Adım {i:5d}/{steps} | Train Loss: {loss.item():.4f}')

# 5. DEĞERLENDİRME VE ÖRNEKLEME

for layer in model.layers:
  if hasattr(layer, 'training'):
    layer.training = False

with torch.no_grad():
  dev_loss_wavenet = F.cross_entropy(model(Xdev), Ydev).item()
  tr_loss_wavenet = F.cross_entropy(model(Xtr), Ytr).item()


def sample_wavenet(num_samples=15, seed=42):
  g_sample = torch.Generator().manual_seed(seed)
  samples = []
  for _ in range(num_samples):
    out = []
    context = [0] * block_size
    while True:
      logits = model(torch.tensor([context]))
      probs = F.softmax(logits, dim=-1)
      ix = torch.multinomial(probs, num_samples=1, generator=g_sample).item()
      context = context[1:] + [ix]
      if ix == 0:
        break
      out.append(itos[ix])
    samples.append(''.join(out))
  return samples


wavenet_names = sample_wavenet(15)

# 6. ÜRETİLEN TÜRKÇE KELİMELERİ / İSİMLERİ YAZDIRMA

print("\n" + "=" * 45)
print("WAVENET TARAFINDAN ÜRETİLEN TÜRKÇE KELİMELER")
print("=" * 45)

wavenet_names = sample_wavenet(num_samples=20, seed=42)

for i, name in enumerate(wavenet_names, 1):
  print(f"{i:2d}. {name}")
print("=" * 45)