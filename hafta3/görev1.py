import os
import ssl
import urllib.request
import torch


ssl_context = ssl._create_unverified_context()

if not os.path.exists('names.txt'):
  url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
  with urllib.request.urlopen(url, context=ssl_context) as response, open(
      'names.txt', 'wb'
  ) as f:
    f.write(response.read())

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
chars = ['.'] + chars
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}

# 27x27 PyTorch Tensörü ile Bigram Sayımları
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1


P = N.float()
P /= P.sum(1, keepdim=True)


g = torch.Generator().manual_seed(2147483647)

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