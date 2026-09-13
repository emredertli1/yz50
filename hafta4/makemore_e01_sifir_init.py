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

n_embd = 10
n_hidden = 200


# ============================================================
# EGZERSİZ E01: TÜM AĞIRLIK VE BIAS'LARI SIFIRLA BAŞLATMAK
# ============================================================
print("=== E01: Sıfır başlangıçla eğitim ===\n")

g = torch.Generator().manual_seed(2147483647)

# Hepsi sıfır! (embedding tablosu C de dahil)
C = torch.zeros((vocab_size, n_embd))
W1 = torch.zeros((n_embd * block_size, n_hidden))
b1 = torch.zeros(n_hidden)
W2 = torch.zeros((n_hidden, vocab_size))
b2 = torch.zeros(vocab_size)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

print(f"Başlangıçta tüm parametreler sıfır mı? {all((p == 0).all().item() for p in parameters)}\n")

# --- İlk forward pass'e bakalım, çökme öncesi durum ---
Xb, Yb = Xtr[:32], Ytr[:32]
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hpreact = embcat @ W1 + b1
h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yb)

print(f"İlk loss: {loss.item():.4f}")
print(f"hpreact'in tamamı sıfır mı? {(hpreact == 0).all().item()}")
print(f"h'nin tamamı sıfır mı (tanh(0)=0 olduğu için)? {(h == 0).all().item()}")
print(f"logits'in tamamı sıfır mı? {(logits == 0).all().item()}")
print("-> Her şey sıfır olduğu için, softmax tüm sınıflara EŞİT olasılık veriyor.")
print(f"   Bu yüzden loss = -ln(1/{vocab_size}) = {torch.log(torch.tensor(float(vocab_size))).item():.4f} civarı, ki gördüğümüz de bu.\n")

# --- İlk backward pass'e bakalım: gradyanlar nasıl görünüyor? ---
loss.backward()

print("=== Gradyan analizi (ilk adım) ===")
print(f"C.grad tamamen sıfır mı? {(C.grad == 0).all().item()}")
print(f"W1.grad tamamen sıfır mı? {(W1.grad == 0).all().item()}")
print(f"b1.grad tamamen sıfır mı? {(b1.grad == 0).all().item()}")
print(f"W2.grad tamamen sıfır mı? {(W2.grad == 0).all().item()}")
print(f"b2.grad tamamen sıfır mı? {(b2.grad == 0).all().item()}\n")

print("W2.grad'ın satırlarına bakalım (her satır birbirinden farklı mı?):")
print(W2.grad[0][:5])
print(W2.grad[1][:5])
print("-> W2.grad'ın satırları FARKLI, çünkü b2'nin gradyanı Y'nin gerçek dağılımına bağlı")
print("   (hangi harfin ne sıklıkla göründüğüne göre farklılaşıyor) - bu yüzden b2 ve W2 kısmen öğrenebiliyor.\n")

print("W1.grad'ın satırlarına bakalım:")
print(W1.grad[0][:5])
print(W1.grad[1][:5])
print("-> W1.grad'ın TÜM satırları birbirinin AYNISI (ya da çok yakın), çünkü h tamamen sıfır olduğu için")
print("   W1'e giden gradyan da simetrik kalıyor - hiçbir gizli nöron diğerinden 'farklılaşamıyor'.\n")

print("C.grad'a bakalım:")
print(C.grad[1][:5])
print(C.grad[5][:5])
print("-> Aynı simetri sorunu C için de geçerli, tüm harflerin embedding gradyanı birbirine çok benzer.\n")


# --- Şimdi bu haliyle biraz eğitip ne olduğunu gözlemleyelim ---
print("=== 2000 adım eğitim sonrası durum ===\n")

lr = 0.1
for i in range(2000):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)

    emb = C[Xtr[ix]]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    for p in parameters:
        p.data += -lr * p.grad

print(f"2000 adım sonrası loss: {loss.item():.4f}\n")

# --- Şimdi W1'in satırlarının hâlâ simetrik olup olmadığına bakalım ---
print("W1'in ilk 3 satırı hâlâ birbirine çok yakın mı?")
print("Satır 0 - Satır 1 arasındaki fark (ortalama mutlak):", (W1[0] - W1[1]).abs().mean().item())
print("Satır 0 - Satır 2 arasındaki fark (ortalama mutlak):", (W1[0] - W1[2]).abs().mean().item())
print("-> Bu farklar hâlâ çok küçükse, W1'in nöronları birbirinden yeterince ayrışamamış demektir")
print("   (b1 tamamen sıfır kaldığı, ve C de simetrik başladığı için, W1'in satırları birbirine çok bağımlı kalıyor).\n")

# --- C'nin (embedding) durumuna bakalım ---
print("C'nin satırları arasında fark oluştu mu?")
print("C[1] (bir harf) - C[5] (başka harf) arasındaki fark:", (C[1] - C[5]).abs().mean().item())
print("-> Eğer bu sıfırdan farklıysa, C kısmen öğrenebilmiş demektir - çünkü C'ye gelen gradyan,")
print("   hangi harfin context'te nerede göründüğüne bağlı olarak farklılaşabiliyor (W1 tamamen simetrik olsa bile,")
print("   farklı harfler farklı Y dağılımlarına yol açtığı için C tam simetrik kalamıyor).\n")

# --- Karşılaştırma için normal (rastgele) başlangıçla eğitilmiş modelin durumunu hatırlatalım ---
print("=== SONUÇ / GENEL YORUM ===")
print("1. Çıkış katmanı (W2, b2) kısmen öğrenebiliyor - çünkü onlara giden gradyan,")
print("   doğrudan gerçek harf dağılımına (Y) bağlı, simetriye o kadar bağımlı değil.")
print("2. Embedding tablosu (C) da kısmen öğrenebiliyor - farklı harfler farklı bağlamlarda")
print("   farklı Y dağılımlarına yol açtığı için, C'nin satırları arasında ufak da olsa fark oluşuyor.")
print("3. Gizli katman (W1, b1) neredeyse HİÇ öğrenemiyor - çünkü tüm nöronlar aynı hesaplamayı yapıyor")
print("   (aynı ağırlıkla başladıkları ve aynı gradyanı aldıkları için), yani '100 nöron' yerine")
print("   pratikte '1 nöron 100 kere kopyalanmış' gibi davranıyor. Bu yüzden loss, rastgele başlangıca")
print("   göre çok daha yavaş düşüyor ve belirli bir noktadan sonra iyileşmiyor.")
print("-> Bu, rastgele başlangıcın (random init) neden gerekli olduğunu somut olarak gösteriyor:")
print("   sıfır başlangıç, 'symmetry breaking' sorununa yol açıyor, ağın kapasitesinin büyük kısmını")
print("   kullanılamaz hale getiriyor.")