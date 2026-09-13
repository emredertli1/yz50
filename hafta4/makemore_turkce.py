import torch
import torch.nn.functional as F
import random

# ============================================================
# TÜRKÇE VERİ SETİNİ OKUMA
# ============================================================
words = open('turkish_names.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower() for w in words]  # büyük/küçük harf tutarlılığı için

chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
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

print(f"Toplam kelime sayısı: {len(words)}")
print(f"Alfabe boyutu (Türkçe karakterler dahil): {vocab_size}")
print(f"Kullanılan karakterler: {chars}\n")

n_embd = 10
n_hidden = 200


# ============================================================
# MLP MODELİ (embedding + gizli katman + batchnorm)
# ============================================================
def train_mlp(n_steps=30000, batch_size=32, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=g)

    fan_in1 = n_embd * block_size
    W1 = torch.randn((fan_in1, n_hidden), generator=g) * (5/3) / (fan_in1**0.5)
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    b2 = torch.randn(vocab_size, generator=g) * 0

    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))
    bnmean_running = torch.zeros((1, n_hidden))
    bnstd_running = torch.ones((1, n_hidden))

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True

    lr = 0.1
    for i in range(n_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

        emb = C[Xtr[ix]]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ W1 + b1

        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])

        for p in parameters:
            p.grad = None
        loss.backward()

        if i > 20000:
            lr = 0.01
        for p in parameters:
            p.data += -lr * p.grad

        if i % 5000 == 0:
            print(f"iter {i}: train loss {loss.item():.4f}")

    return C, W1, b1, W2, b2, bngain, bnbias, bnmean_running, bnstd_running


def eval_mlp(params, X, Y):
    C, W1, b1, W2, b2, bngain, bnbias, bnmean_running, bnstd_running = params
    emb = C[X]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()


def sample_mlp(params, n_samples=20, seed=2147483647):
    C, W1, b1, W2, b2, bngain, bnbias, bnmean_running, bnstd_running = params
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n_samples):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            embcat = emb.view(1, -1)
            hpreact = embcat @ W1 + b1
            hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
            h = torch.tanh(hpreact)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(itos[ix])
            if ix == 0:
                break
        names.append(''.join(out[:-1]))
    return names


# ============================================================
# BIGRAM MODELİ (karşılaştırma için)
# ============================================================
def train_bigram():
    xs_b, ys_b = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs_b.append(stoi[ch1])
            ys_b.append(stoi[ch2])
    xs_b = torch.tensor(xs_b)
    ys_b = torch.tensor(ys_b)

    g = torch.Generator().manual_seed(2147483647)
    Wb = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

    for k in range(200):
        xenc = F.one_hot(xs_b, num_classes=vocab_size).float()
        logits = xenc @ Wb
        loss = F.cross_entropy(logits, ys_b)

        Wb.grad = None
        loss.backward()
        with torch.no_grad():
            Wb -= 50 * Wb.grad

    return Wb, loss.item()


def sample_bigram(Wb, n_samples=20, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n_samples):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
            logits = xenc @ Wb
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        names.append(''.join(out))
    return names


# ============================================================
# ÇALIŞTIRMA VE KARŞILAŞTIRMA
# ============================================================
print("=== MLP (trigram + embedding + batchnorm) eğitiliyor ===")
mlp_params = train_mlp()
mlp_dev_loss = eval_mlp(mlp_params, Xdev, Ydev)

print("\n=== Bigram modeli eğitiliyor ===")
Wb, bigram_train_loss = train_bigram()

print("\n" + "="*60)
print(f"{'MODEL':<15} {'DEV/TRAIN LOSS':<20}")
print("="*60)
print(f"{'MLP':<15} {mlp_dev_loss:<20.4f} (dev loss)")
print(f"{'Bigram':<15} {bigram_train_loss:<20.4f} (train loss, dev ayrımı yok)")

print("\n" + "="*60)
print("TÜRKÇE İSİM ÖRNEKLERİ - YAN YANA KARŞILAŞTIRMA")
print("="*60)

mlp_names = sample_mlp(mlp_params)
bigram_names = sample_bigram(Wb)

print(f"{'MLP (trigram)':<25} {'Bigram':<25}")
print("-"*50)
for m, b in zip(mlp_names, bigram_names):
    print(f"{m:<25} {b:<25}")

print("\n=== YORUM ===")
print("MLP modeli son 3 harfe baktığı ve harfleri embedding ile temsil ettiği için,")
print("Türkçe'ye özgü harf kombinasyonlarını (ör. 'ş', 'ç', 'ğ' geçişleri) bigram'a göre")
print("daha tutarlı yakalamalı ve daha 'isim gibi' çıktılar üretmeli.")
print("Bigram ise sadece bir önceki harfe baktığı için daha rastgele/parçalı sonuçlar verir.")