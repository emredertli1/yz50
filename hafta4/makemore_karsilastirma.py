import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# ============================================================
# VERİ HAZIRLIĞI
# ============================================================
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
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


# ============================================================
# MODEL EĞİTME FONKSİYONU (farklı boyutlarla tekrar tekrar kullanmak için)
# ============================================================
def train_model(n_embd, n_hidden, n_steps=30000, batch_size=32, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((27, n_embd), generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * 0.2
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    W2 = torch.randn((n_hidden, 27), generator=g) * 0.01
    b2 = torch.randn(27, generator=g) * 0
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    lr = 0.1
    for i in range(n_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

        emb = C[Xtr[ix]]
        embcat = emb.view(emb.shape[0], -1)
        h = torch.tanh(embcat @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])

        for p in parameters:
            p.grad = None
        loss.backward()

        if i > 20000:
            lr = 0.01

        for p in parameters:
            p.data += -lr * p.grad

    # dev loss'u hesapla
    emb = C[Xdev]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    dev_loss = F.cross_entropy(logits, Ydev).item()

    return parameters, dev_loss


# ============================================================
# ADIM 1: Gizli katman ve embedding boyutunu büyütüp karşılaştırma
# ============================================================
print("=== Farklı boyutlarla dev loss karşılaştırması ===")

configs = [
    (2, 100),   # küçük (geçen haftaki gibi)
    (10, 100),  # embedding büyüdü
    (10, 200),  # gizli katman da büyüdü
    (20, 300),  # ikisi de büyük
]

results = {}
for n_embd, n_hidden in configs:
    params, dev_loss = train_model(n_embd, n_hidden)
    results[(n_embd, n_hidden)] = (params, dev_loss)
    print(f"n_embd={n_embd}, n_hidden={n_hidden} -> dev loss: {dev_loss:.4f}")


# ============================================================
# ADIM 2: Embedding'leri 2 boyutta çizdirmek
# ============================================================
# Görselleştirme için embedding boyutu 2 olan modeli kullanıyoruz
C_vis = results[(2, 100)][0][0]  # bu config'in C tablosu

plt.figure(figsize=(8, 8))
plt.scatter(C_vis[:, 0].data, C_vis[:, 1].data, s=200)
for i in range(C_vis.shape[0]):
    plt.text(C_vis[i, 0].item(), C_vis[i, 1].item(), itos[i],
              ha="center", va="center", color='white')
plt.grid('minor')
plt.title("Harf embedding'leri (2 boyutlu)")
plt.savefig('embeddings.png')
plt.show()

print("\nGrafik 'embeddings.png' olarak kaydedildi.")
print("Grafikte birbirine yakın düşen harfler, modele göre isimlerde benzer şekilde davranan harfler.")


# ============================================================
# ADIM 3: En iyi modelden isim örneklemek
# ============================================================
def sample_names(parameters, n_embd, n_hidden, n_samples=20, seed=2147483647):
    C, W1, b1, W2, b2 = parameters
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n_samples):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            embcat = emb.view(1, -1)
            h = torch.tanh(embcat @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(itos[ix])
            if ix == 0:
                break
        names.append(''.join(out[:-1]))
    return names

# en iyi dev loss'a sahip modeli seç
best_config = min(results, key=lambda k: results[k][1])
best_params = results[best_config][0]
print(f"\n=== En iyi model (n_embd={best_config[0]}, n_hidden={best_config[1]}) ile üretilen isimler ===")
mlp_names = sample_names(best_params, *best_config)
for name in mlp_names:
    print(name)


# ============================================================
# ADIM 4: Bigram modeliyle karşılaştırma
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
    num = xs_b.nelement()

    g = torch.Generator().manual_seed(2147483647)
    Wb = torch.randn((27, 27), generator=g, requires_grad=True)

    for k in range(200):
        xenc = F.one_hot(xs_b, num_classes=27).float()
        logits = xenc @ Wb
        loss = F.cross_entropy(logits, ys_b)

        Wb.grad = None
        loss.backward()
        with torch.no_grad():
            Wb -= 50 * Wb.grad

    return Wb

def sample_bigram(Wb, n_samples=20, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(n_samples):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ Wb
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        names.append(''.join(out))
    return names

print("\n=== Bigram modeli ile üretilen isimler (karşılaştırma) ===")
Wb = train_bigram()
bigram_names = sample_bigram(Wb)
for name in bigram_names:
    print(name)

print("\n=== Karşılaştırma özeti ===")
print("MLP (trigram + embedding) isimleri daha 'isim gibi' görünmeli,")
print("çünkü model son 3 harfe bakıyor, bigram sadece son 1 harfe bakıyor.")