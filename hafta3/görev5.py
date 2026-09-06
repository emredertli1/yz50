import os
import ssl
import urllib.request
import torch
import torch.nn.functional as F

ssl_context = ssl._create_unverified_context()
url = 'https://gist.githubusercontent.com/sukruozan/225b36b5cb4480a619f726aff455afea/raw/tr_long_name_list.txt'

if not os.path.exists('turkish_names.txt'):
  with urllib.request.urlopen(url, context=ssl_context) as response, open(
      'turkish_names.txt', 'wb'
  ) as f:
    f.write(response.read())

words = open('turkish_names.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.strip().lower() for w in words if w.strip()]

chars = sorted(
    list(
        set(
            ''.join(words)
            + 'abcçdefgğhıijklmnoöprsştuüvyz'
        )
    )
)
chars = ['.'] + chars
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}

N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    N[stoi[ch1], stoi[ch2]] += 1

P = (N + 1).float()
P /= P.sum(1, keepdim=True)

log_likelihood = 0.0
n = 0
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    prob = P[stoi[ch1], stoi[ch2]]
    log_likelihood += torch.log(prob)
    n += 1
nll_count = -log_likelihood / n
print(f'Sayım Modeli NLL Loss: {nll_count.item():.4f}')

g = torch.Generator().manual_seed(2147483647)
print('\nSayım Modeli Örnekleri:')
for i in range(10):
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(
        p, num_samples=1, replacement=True, generator=g
    ).item()
    if ix == 0:
      break
    out.append(itos[ix])
  print(''.join(out))

xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    xs.append(stoi[ch1])
    ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

for k in range(300):
  xenc = F.one_hot(xs, num_classes=vocab_size).float()
  logits = xenc @ W
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdim=True)
  loss = (
      -probs[torch.arange(num), ys].log().mean()
      + 0.01 * (W ** 2).mean()
  )

  W.grad = None
  loss.backward()

  with torch.no_grad():
    W -= 50 * W.grad

print(f'\nSinir Ağı Modeli Final Loss: {loss.item():.4f}')