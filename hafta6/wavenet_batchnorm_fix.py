import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 1. VERİ HAZIRLIĞI (block_size = 8)

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

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

vocab_size = 27
n_embd = 24
n_hidden = 128


# ŞEKİLLERİ YAZDIRIP HATAYI GÖSTERME

print('--- ŞEKİL HATASI ANALİZİ ---')
dummy_x = torch.randn(32, 4, 68)

mean_faulty = dummy_x.mean(0, keepdim=True)
mean_correct = dummy_x.mean((0, 1), keepdim=True)

print(f'Girdi Tensör Boyutu (B, T, C)          : {dummy_x.shape}')
print(f'Hatalı Ortalama x.mean(0) Boyutu       : {mean_faulty.shape}  <-- Yanlış!')
print(
    f'Düzeltilmiş Ortalama x.mean((0, 1))     : {mean_correct.shape}  <--'
    ' Doğru (Kanal başına 1 istatistik)\n'
)

# 3. TEMEL KATMANLAR

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

# 4. BATCHNORM SINIFLARI: HATALI vs DÜZELTİLMİŞ

class BatchNorm1dFaulty:

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
        ) * self.running_mean + self.momentum * xmean.mean(1).squeeze(0)
        self.running_var = (
            1 - self.momentum
        ) * self.running_var + self.momentum * xvar.mean(1).squeeze(0)

    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

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

# 5. MODEL OLUŞTURMA, EĞİTİM VE ÖRNEKLEME MOTORU

def create_model(bn_class, seed=42):
  torch.manual_seed(seed)
  model = Sequential([
      Embedding(vocab_size, n_embd),  # (B, 8, 24)
      FlattenConsecutive(2),  # (B, 4, 48)
      Linear(n_embd * 2, n_hidden, bias=False),  # (B, 4, 128)
      bn_class(n_hidden),
      Tanh(),
      FlattenConsecutive(2),  # (B, 2, 256)
      Linear(n_hidden * 2, n_hidden, bias=False),  # (B, 2, 128)
      bn_class(n_hidden),
      Tanh(),
      FlattenConsecutive(2),  # (B, 128)
      Linear(n_hidden * 2, n_hidden, bias=False),  # (B, 128)
      bn_class(n_hidden),
      Tanh(),
      Linear(n_hidden, vocab_size),  # (B, 27)
  ])
  with torch.no_grad():
    model.layers[-1].weight *= 0.1

  for p in model.parameters():
    p.requires_grad = True
  return model


def train_and_evaluate(model, steps=20000, batch_size=32):
  parameters = model.parameters()
  g = torch.Generator().manual_seed(42)
  lossi = []

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

    lossi.append(loss.item())

  # Eval kipine geçiş
  for layer in model.layers:
    if hasattr(layer, 'training'):
      layer.training = False

  with torch.no_grad():
    logits_dev = model(Xdev)
    dev_loss = F.cross_entropy(logits_dev, Ydev).item()

    logits_tr = model(Xtr)
    tr_loss = F.cross_entropy(logits_tr, Ytr).item()

  return tr_loss, dev_loss, lossi


def sample_names(model, num_samples=5, seed=42):
  """Eğitilmiş modelden yeni isimler üretir (Sampling)."""
  g = torch.Generator().manual_seed(seed)
  generated = []

  for _ in range(num_samples):
    out = []
    context = [0] * block_size
    while True:
      # (1, block_size) tensör ile ileri yayılım
      logits = model(torch.tensor([context]))
      probs = F.softmax(logits, dim=-1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      if ix == 0:
        break
      out.append(itos[ix])
    generated.append(''.join(out))

  return generated

# 6. DENEYİ ÇALIŞTIRMA VE KARŞILAŞTIRMA

print('1) Hatalı BatchNorm1d ile model eğitiliyor...')
model_faulty = create_model(BatchNorm1dFaulty)
tr_loss_faulty, dev_loss_faulty, lossi_faulty = train_and_evaluate(model_faulty)

print('2) Düzeltilmiş BatchNorm1d ile model eğitiliyor...')
model_correct = create_model(BatchNorm1dCorrect)
tr_loss_correct, dev_loss_correct, lossi_correct = train_and_evaluate(
    model_correct
)

# Metrik Tablosu
print('\n' + '=' * 48)
print(f"{'BatchNorm Yapısı':<25} {'Train Loss':<11} {'Dev Loss':<11}")
print('-' * 48)
print(
    f"{'Düzeltme ÖNCESİ (x.mean(0))':<25} {tr_loss_faulty:<11.4f}"
    f' {dev_loss_faulty:<11.4f}'
)
print(
    f"{'Düzeltme SONRASI ((0, 1))':<25} {tr_loss_correct:<11.4f}"
    f' {dev_loss_correct:<11.4f}'
)
print('=' * 48)

# Üretilen İsim Örnekleri
print('\n--- MODEL ÇIKTI ÖRNEKLERİ (SAMPLING) ---')
print('Hatalı Modelden İsimler:')
for name in sample_names(model_faulty, num_samples=5):
  print(f'  - {name}')

print('\nDüzeltilmiş Modelden İsimler:')
for name in sample_names(model_correct, num_samples=5):
  print(f'  - {name}')

# 7. LOSS GRAFİĞİNİ ÇİZDİRME (Moving Average)

# 20.000 adımı 200'lük pencerelerle yumuşatarak görselleştirme
window_size = 200
faulty_smooth = (
    torch.tensor(lossi_faulty).view(-1, window_size).mean(dim=1).numpy()
)
correct_smooth = (
    torch.tensor(lossi_correct).view(-1, window_size).mean(dim=1).numpy()
)

plt.figure(figsize=(10, 5))
plt.plot(faulty_smooth, label='Hatalı BatchNorm1d: x.mean(0)', color='red')
plt.plot(correct_smooth, label='Düzeltilmiş BatchNorm1d: x.mean((0, 1))', color='green')
plt.title('WaveNet: BatchNorm Ekseni Düzeltmesi Öncesi ve Sonrası Eğitim Kaybı')
plt.xlabel(f'Adım x {window_size}')
plt.ylabel('Loss (Cross Entropy)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()