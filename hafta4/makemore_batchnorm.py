import torch
import torch.nn.functional as F
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

n_embd = 10
n_hidden = 200
vocab_size = 27


# ============================================================
# MODEL 1: BATCHNORM'SUZ (KARŞILAŞTIRMA İÇİN TEMEL MODEL)
# ============================================================
def train_without_batchnorm(n_steps=30000, batch_size=32, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=g)

    fan_in1 = n_embd * block_size
    W1 = torch.randn((fan_in1, n_hidden), generator=g) * (5/3) / (fan_in1**0.5)
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    b2 = torch.randn(vocab_size, generator=g) * 0

    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    lr = 0.1
    for i in range(n_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

        emb = C[Xtr[ix]]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ W1 + b1
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

    return C, W1, b1, W2, b2


def eval_without_batchnorm(params, X, Y):
    C, W1, b1, W2, b2 = params
    emb = C[X]
    embcat = emb.view(emb.shape[0], -1)
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()


# ============================================================
# MODEL 2: BATCHNORM'LU
# ============================================================
def train_with_batchnorm(n_steps=30000, batch_size=32, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=g)

    fan_in1 = n_embd * block_size
    W1 = torch.randn((fan_in1, n_hidden), generator=g) * (5/3) / (fan_in1**0.5)
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    b2 = torch.randn(vocab_size, generator=g) * 0

    # --- BatchNorm parametreleri ---
    bngain = torch.ones((1, n_hidden))    # öğrenilebilir ölçek
    bnbias = torch.zeros((1, n_hidden))   # öğrenilebilir kaydırma

    # --- Tahmin (inference) zamanı için "running" istatistikler ---
    # Bunlar eğitilmiyor (requires_grad yok), eğitim sırasında yavaşça güncelleniyor
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

        # --- BatchNorm: EĞİTİM sırasında batch istatistiğini kullan ---
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

        # --- running mean/std'yi güncelle (gradyan takibi olmadan) ---
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

    return C, W1, b1, W2, b2, bngain, bnbias, bnmean_running, bnstd_running


def eval_with_batchnorm(params, X, Y):
    C, W1, b1, W2, b2, bngain, bnbias, bnmean_running, bnstd_running = params
    emb = C[X]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1

    # --- BatchNorm: TAHMİN sırasında running mean/std kullan ---
    # Neden batch istatistiği değil? Çünkü tahminde tek bir örnek olabilir,
    # tek örneğin "std"si anlamsız/sıfır olur. Bu yüzden eğitim boyunca biriktirdiğimiz
    # running istatistikleri kullanıyoruz - böylece batch boyutundan bağımsız çalışabiliyoruz.
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias

    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()


# ============================================================
# KARŞILAŞTIRMA
# ============================================================
print("=== Eğitim başlıyor ===\n")

print("1) BatchNorm'suz model eğitiliyor...")
params_no_bn = train_without_batchnorm()
dev_loss_no_bn = eval_without_batchnorm(params_no_bn, Xdev, Ydev)
train_loss_no_bn = eval_without_batchnorm(params_no_bn, Xtr, Ytr)

print("2) BatchNorm'lu model eğitiliyor...")
params_bn = train_with_batchnorm()
dev_loss_bn = eval_with_batchnorm(params_bn, Xdev, Ydev)
train_loss_bn = eval_with_batchnorm(params_bn, Xtr, Ytr)

print("\n=== SONUÇLAR ===")
print(f"{'Model':<20} {'Train Loss':<12} {'Dev Loss':<12}")
print(f"{'BatchNorm YOK':<20} {train_loss_no_bn:<12.4f} {dev_loss_no_bn:<12.4f}")
print(f"{'BatchNorm VAR':<20} {train_loss_bn:<12.4f} {dev_loss_bn:<12.4f}")

print("\n=== YORUM ===")
print("BatchNorm, her mini-batch'te gizli katmanın çıktısını (hpreact) normalize ediyor:")
print("- Eğitim sırasında: o anki batch'in kendi ortalaması ve standart sapması kullanılıyor.")
print("- Tahmin sırasında: eğitim boyunca biriktirilen 'running mean/std' kullanılıyor,")
print("  çünkü tahminde tek bir örnek gelebilir ve onun kendi istatistiği güvenilir olmaz.")
print("BatchNorm sayesinde hpreact'in dağılımı eğitim boyunca stabil kalıyor,")
print("bu da Part 3'te gördüğümüz 'tanh doyması' sorununu daha da azaltıyor")
print("ve genelde biraz daha iyi (daha düşük) dev loss ile sonuçlanıyor.")