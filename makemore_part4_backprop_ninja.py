import torch
import torch.nn.functional as F

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

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

n_embd = 10    
n_hidden = 64   

g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd),            generator=g)

W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 

bngain = torch.randn((1, n_hidden),               generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden),               generator=g) * 0.1

W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True


n = 32
Xb, Yb = Xtr[:n], Ytr[:n]


emb = C[Xb]                                     
embcat = emb.view(emb.shape[0], -1)         

hprebn = embcat @ W1 + b1                      

bnmeani = 1/n * hprebn.sum(0, keepdim=True)     
bndiff = hprebn - bnmeani                      
bndiff2 = bndiff**2                               
bnvar = 1/(n - 1) * bndiff2.sum(0, keepdim=True)  
bnvar_inv = (bnvar + 1e-5)**-0.5                 
bnraw = bndiff * bnvar_inv                    
hpreact = bngain * bnraw + bnbias            

h = torch.tanh(hpreact)                        

logits = h @ W2 + b2                       

logit_maxes = logits.max(1, keepdim=True).values  
norm_logits = logits - logit_maxes               
counts = norm_logits.exp()                        
counts_sum = counts.sum(1, keepdim=True)       
counts_sum_inv = counts_sum**-1               
probs = counts * counts_sum_inv             
logprobs = probs.log()                           
loss = -logprobs[range(n), Yb].mean()

for t in [emb, embcat, hprebn, bnmeani, bndiff, bndiff2, bnvar, bnvar_inv, 
          bnraw, hpreact, h, logits, logit_maxes, norm_logits, counts, 
          counts_sum, counts_sum_inv, probs, logprobs]:
    t.retain_grad()

for p in parameters:
    p.grad = None

loss.backward()

def cmp(s, dt, t):
    exact = torch.equal(dt, t.grad)
    approx = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f"{s:15s} | EXACT: {str(exact):5s} | APPROX: {str(approx):5s} | MAX DIFF: {maxdiff:.2e}")


#  MANUEL GERİYE YAYILIM (Backpropagation by Hand)

print("=== ARA DEĞİŞKENLERİN KARŞILAŞTIRILMASI ===")

# dloss / dlogprobs
# loss = - (1/n) * sum(logprobs[i, Yb[i]])
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n
cmp('logprobs', dlogprobs, logprobs)

# dloss / dprobs
# logprobs = probs.log() => d/dprobs = 1 / probs
dprobs = (1.0 / probs) * dlogprobs
cmp('probs', dprobs, probs)

# probs = counts * counts_sum_inv
# Burada iki dala ayrılıyor: dcounts_sum_inv ve dcounts (1. parça)
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

# counts_sum_inv = counts_sum**-1 => d = -counts_sum**-2
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)

# counts_sum = counts.sum(1, keepdim=True)
# counts iki yerden gradyan alır: (1) probs çarpımından, (2) counts_sum toplamından
dcounts = counts_sum_inv * dprobs + torch.ones_like(counts) * dcounts_sum
cmp('counts', dcounts, counts)

# counts = norm_logits.exp() => d = exp(norm_logits) = counts
dnorm_logits = counts * dcounts
cmp('norm_logits', dnorm_logits, norm_logits)

# norm_logits = logits - logit_maxes
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
cmp('logit_maxes', dlogit_maxes, logit_maxes)

# logits iki yerden gradyan alır: norm_logits ve logit_maxes (max indeksi)
dlogits = dnorm_logits.clone()
dlogits += F.one_hot(logits.max(1).indices, num_classes=vocab_size) * dlogit_maxes
cmp('logits', dlogits, logits)

# logits = h @ W2 + b2
dh = dlogits @ W2.T
cmp('h', dh, h)

dW2 = h.T @ dlogits
cmp('W2', dW2, W2)

db2 = dlogits.sum(0)
cmp('b2', db2, b2)

# h = tanh(hpreact) => dhpreact = (1 - tanh^2) * dh
dhpreact = (1.0 - h**2) * dh
cmp('hpreact', dhpreact, hpreact)

# hpreact = bngain * bnraw + bnbias
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
cmp('bngain', dbngain, bngain)

dbnbias = dhpreact.sum(0, keepdim=True)
cmp('bnbias', dbnbias, bnbias)

dbnraw = bngain * dhpreact
cmp('bnraw', dbnraw, bnraw)

# bnraw = bndiff * bnvar_inv
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

# bnvar_inv = (bnvar + 1e-5)**-0.5
dbnvar = (-0.5 * (bnvar + 1e-5)**-1.5) * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)

# bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)
dbndiff2 = (1.0 / (n - 1)) * torch.ones_like(bndiff2) * dbnvar
cmp('bndiff2', dbndiff2, bndiff2)

# bndiff2 = bndiff**2 ve bndiff iki koldan gelir
dbndiff = (2.0 * bndiff) * dbndiff2 + bnvar_inv * dbnraw
cmp('bndiff', dbndiff, bndiff)

# bndiff = hprebn - bnmeani
dbnmeani = (-dbndiff).sum(0, keepdim=True)
cmp('bnmeani', dbnmeani, bnmeani)

# hprebn = embcat @ W1 + b1 (iki koldan gradyan alır: bndiff ve bnmeani)
dhprebn = dbndiff.clone() + (1.0 / n) * torch.ones_like(hprebn) * dbnmeani
cmp('hprebn', dhprebn, hprebn)

# hprebn = embcat @ W1 + b1
dW1 = embcat.T @ dhprebn
cmp('W1', dW1, W1)

db1 = dhprebn.sum(0)
cmp('b1', db1, b1)

dembcat = dhprebn @ W1.T
cmp('embcat', dembcat, embcat)

# embcat = emb.view(emb.shape[0], -1)
demb = dembcat.view(emb.shape)
cmp('emb', demb, emb)

# emb = C[Xb]
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]
cmp('C', dC, C)

print("\nTEBRİKLER! TÜM TÜREVLER BAŞARIYLA EŞLEŞTİ!")