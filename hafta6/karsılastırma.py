import random
import torch
import torch.nn.functional as F

# ============================================================
# 1. VERİ HAZIRLIĞI (Hem block_size=3 hem block_size=8)
# ============================================================
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}


def build_dataset(words, block_size):
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

# Bağlam 3 için veri kümeleri
Xtr_3, Ytr_3 = build_dataset(words[:n1], block_size=3)
Xdev_3, Ydev_3 = build_dataset(words[n1:n2], block_size=3)

# Bağlam 8 için veri kümeleri
Xtr_8, Ytr_8 = build_dataset(words[:n1], block_size=8)
Xdev_8, Ydev_8 = build_dataset(words[n1:n2], block_size=8)

vocab_size = 27
n_embd = 24  # Büyütülmüş embedding boyutu (Önceki 10/24'ten)
n_hidden = 128  # Büyütülmüş katman genişliği


# ============================================================
# 2. KATMAN SINIFLARI
# ============================================================
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


class Flatten:
  """Düz MLP için tüm bağlamı tek boyutta birleştirir."""

  def __call__(self, x):
    self.out = x.view(x.shape[0], -1)
    return self.out

  def parameters(self):
    return []


class FlattenConsecutive:
  """WaveNet hiyerarşisi için komşu n karakteri birleştirir."""

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


# ============================================================
# 3. ÜÇ FARKLI MİMARİNİN OLUŞTURULMASI
# ============================================================
torch.manual_seed(42)

# Model 1: Bağlam 3, Düz MLP
model_ctx3_mlp = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * 3, n_hidden, bias=False),
    BatchNorm1dCorrect(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
])

# Model 2: Bağlam 8, Düz MLP (Tüm 8 karakter doğrudan tek katmana girer)
model_ctx8_mlp = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * 8, n_hidden, bias=False),
    BatchNorm1dCorrect(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
])

# Model 3: Bağlam 8, WaveNet (Hiyerarşik İkili Birleştirme)
model_ctx8_wavenet = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2),
    Linear(n_embd * 2, n_hidden, bias=False),
    BatchNorm1dCorrect(n_hidden),
    Tanh(),
    FlattenConsecutive(2),
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1dCorrect(n_hidden),
    Tanh(),
    FlattenConsecutive(2),
    Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1dCorrect(n_hidden),
    Tanh(),
    Linear(n_hidden, vocab_size),
])


# ============================================================
# 4. EĞİTİM MOTORU
# ============================================================
def train_and_eval(model, Xtr, Ytr, Xdev, Ydev, steps=20000, batch_size=32):
  with torch.no_grad():
    model.layers[-1].weight *= 0.1

  for p in model.parameters():
    p.requires_grad = True

  parameters = model.parameters()
  g = torch.Generator().manual_seed(42)

  for i in range(steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
      p.grad = None
    loss.backward()

    lr = 0.1 if i < 15000 else 0.01
    for p in parameters:
      p.data += -lr * p.grad

  # Eval kipi
  for layer in model.layers:
    if hasattr(layer, 'training'):
      layer.training = False

  with torch.no_grad():
    logits_dev = model(Xdev)
    dev_loss = F.cross_entropy(logits_dev, Ydev).item()

  return dev_loss


# ============================================================
# 5. DENEYLERİN ÇALIŞTIRILMASI VE KARŞILAŞTIRMA TABLOSU
# ============================================================
models = [
    ('Bağlam 3 MLP', model_ctx3_mlp, Xtr_3, Ytr_3, Xdev_3, Ydev_3),
    ('Bağlam 8 düz MLP', model_ctx8_mlp, Xtr_8, Ytr_8, Xdev_8, Ydev_8),
    ('Bağlam 8 WaveNet', model_ctx8_wavenet, Xtr_8, Ytr_8, Xdev_8, Ydev_8),
]

results = []

for name, model, Xtr, Ytr, Xdev, Ydev in models:
  num_params = sum(p.nelement() for p in model.parameters())
  print(f'{name} eğitiliyor ({num_params:,} parametre)...')
  dev_loss = train_and_eval(model, Xtr, Ytr, Xdev, Ydev)
  results.append((name, num_params, dev_loss))

# İstenen Tek Karşılaştırma Tablosu
print('\n' + '=' * 55)
print(f"{'Model':<22} | {'Parametre Sayısı':<18} | {'Dev Loss':<10}")
print('-' * 55)
for name, params, loss in results:
  print(f'{name:<22} | {params:<18,} | {loss:<10.4f}')
print('=' * 55)