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

n_embd = 10
n_hidden = 200
vocab_size = 27


# ============================================================
# BÖLÜM A: "KÖTÜ" BAŞLANGIÇ — SORUNU GÖSTERMEK
# ============================================================
print("=== BÖLÜM A: Kötü başlangıçla sorunları gösterme ===\n")

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)   # ÖLÇEKSİZ - sorun burada
b1 = torch.randn(n_hidden, generator=g)                           # ÖLÇEKSİZ
W2 = torch.randn((n_hidden, vocab_size), generator=g)             # ÖLÇEKSİZ
b2 = torch.randn(vocab_size, generator=g)                         # ÖLÇEKSİZ
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# --- 1. Neden başlangıç loss'u çok yüksek? ---
Xb, Yb = Xtr[:32], Ytr[:32]
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hpreact = embcat @ W1 + b1
h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yb)

print(f"Kötü başlangıçtaki ilk loss: {loss.item():.4f}")
print(f"Beklenen 'ideal' başlangıç loss'u: {torch.tensor(1/27.0).log().item()*-1:.4f}  (yani -ln(1/27))")
print("-> Loss çok daha yüksek çünkü W2 ve b2 çok büyük, logits çok büyük/küçük sayılar üretiyor,")
print("   bu da bazı sınıflara aşırı yüksek güven (çok küçük olasılık diğerlerine) veriyor.\n")

# logits'in ne kadar "aşırı" olduğunu gösterelim
print("logits örneği (ilk satır):", logits[0].data)
print("Bu sayılar 0'dan çok uzak, bu yüzden softmax çok keskin bir dağılım üretiyor.\n")

# --- 2. tanh neden doyuyor? histogram çizelim ---
plt.figure(figsize=(10, 4))
plt.hist(hpreact.view(-1).tolist(), 50)
plt.title("hpreact dağılımı (tanh'a girmeden önceki ham değerler) - KÖTÜ BAŞLANGIÇ")
plt.xlabel("değer")
plt.ylabel("frekans")
plt.savefig('hpreact_kotu.png')
plt.show()
print("hpreact histogramı 'hpreact_kotu.png' olarak kaydedildi.")
print("-> hpreact değerleri çok geniş bir aralığa yayılmış (örn -20 ile +20 arası).\n")

plt.figure(figsize=(10, 4))
plt.hist(h.view(-1).tolist(), 50)
plt.title("h dağılımı (tanh SONRASI) - KÖTÜ BAŞLANGIÇ")
plt.xlabel("değer")
plt.ylabel("frekans")
plt.savefig('h_kotu.png')
plt.show()
print("h histogramı 'h_kotu.png' olarak kaydedildi.")
print("-> h'nin büyük çoğunluğu -1 ve +1'e YIĞILMIŞ. Bu 'tanh saturation' (doyma).\n")

# --- Doymuş nöronların oranını sayısal olarak gösterelim ---
saturated = (h.abs() > 0.99).float().mean()
print(f"h'deki değerlerin %{saturated.item()*100:.1f}'i -1 veya +1'e çok yakın (doymuş).")
print("-> Bu doymuş nöronların gradyanı SIFIRA yakın oluyor (tanh'ın türevi 1-tanh^2),")
print("   yani geri yayılımda bu nöronlardan gradyan neredeyse hiç geçmiyor -> öğrenme yavaşlıyor.\n")

# Her nöronun (sütunun) TAMAMEN doymuş olup olmadığını gösteren görsel (beyaz = doymuş)
plt.figure(figsize=(20, 10))
plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')
plt.title("Doymuş nöron haritası (beyaz = |h| > 0.99) - KÖTÜ BAŞLANGIÇ")
plt.savefig('doyma_haritasi_kotu.png')
plt.show()
print("Doyma haritası 'doyma_haritasi_kotu.png' olarak kaydedildi.")
print("Eğer bir SÜTUN tamamen beyazsa, o nöron hiçbir örnekte 'ölmemiş' hale asla dönmüyor demektir - bu 'ölü nöron' riski.\n")


# ============================================================
# BÖLÜM B: KAIMING INIT İLE DÜZELTME
# ============================================================
print("\n=== BÖLÜM B: Kaiming init ile düzeltme ===\n")

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)

# Kaiming init: ağırlıkları girdi boyutunun karekökü ile ölçekliyoruz
# tanh için önerilen kazanç (gain) çarpanı: 5/3
fan_in1 = n_embd * block_size
W1 = torch.randn((fan_in1, n_hidden), generator=g) * (5/3) / (fan_in1**0.5)
b1 = torch.randn(n_hidden, generator=g) * 0.01     # bias'ı küçük tutuyoruz

fan_in2 = n_hidden
W2 = torch.randn((fan_in2, vocab_size), generator=g) * 0.01   # çıkış katmanını küçük başlatıyoruz
b2 = torch.randn(vocab_size, generator=g) * 0

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# --- 1. Yeni başlangıç loss'u ---
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hpreact = embcat @ W1 + b1
h = torch.tanh(hpreact)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yb)

print(f"Kaiming init sonrası ilk loss: {loss.item():.4f}")
print(f"Beklenen 'ideal' başlangıç loss'u: {torch.tensor(1/27.0).log().item()*-1:.4f}")
print("-> Artık çok daha yakın! Çünkü W2 ve b2 küçük olduğu için logits de 0'a yakın,")
print("   yani softmax başlangıçta neredeyse eşit olasılık dağılımı veriyor (mantıklı, çünkü henüz hiçbir şey öğrenmedi).\n")

# --- 2. Yeni hpreact ve h histogramları ---
plt.figure(figsize=(10, 4))
plt.hist(hpreact.view(-1).tolist(), 50)
plt.title("hpreact dağılımı - KAIMING INIT SONRASI")
plt.xlabel("değer")
plt.ylabel("frekans")
plt.savefig('hpreact_iyi.png')
plt.show()
print("hpreact histogramı 'hpreact_iyi.png' olarak kaydedildi.")
print("-> Değerler artık çok daha dar, makul bir aralıkta (örn -2 ile +2 arası).\n")

plt.figure(figsize=(10, 4))
plt.hist(h.view(-1).tolist(), 50)
plt.title("h dağılımı (tanh SONRASI) - KAIMING INIT SONRASI")
plt.xlabel("değer")
plt.ylabel("frekans")
plt.savefig('h_iyi.png')
plt.show()
print("h histogramı 'h_iyi.png' olarak kaydedildi.")
print("-> Değerler artık -1/+1 uçlarına yığılmak yerine, ortada daha dengeli dağılmış.\n")

saturated = (h.abs() > 0.99).float().mean()
print(f"Kaiming init sonrası doymuş nöron oranı: %{saturated.item()*100:.1f}")
print("-> Bu oran, kötü başlangıca göre belirgin şekilde DÜŞTÜ.\n")

plt.figure(figsize=(20, 10))
plt.imshow(h.abs() > 0.99, cmap='gray', interpolation='nearest')
plt.title("Doymuş nöron haritası - KAIMING INIT SONRASI")
plt.savefig('doyma_haritasi_iyi.png')
plt.show()
print("Doyma haritası 'doyma_haritasi_iyi.png' olarak kaydedildi.")
print("-> Beyaz nokta sayısı belirgin şekilde azaldı, tamamen beyaz sütun (ölü nöron) yok.\n")

print("=== ÖZET ===")
print("1. Kötü başlangıçta: loss çok yüksek başlıyordu çünkü çıkış katmanı (W2, b2) çok büyük ölçekteydi.")
print("2. Kötü başlangıçta: tanh, girdileri (hpreact) çok büyük olduğu için -1/+1'e doyuyordu, gradyan akışı bozuluyordu.")
print("3. Kaiming init (W1'i fan_in'e göre ölçekleme) + küçük W2/b2 başlangıcı, her iki sorunu da çözdü:")
print("   - Başlangıç loss'u artık teorik ideale (-ln(1/27)) çok daha yakın.")
print("   - tanh doyma oranı belirgin şekilde düştü, gradyanlar daha sağlıklı akıyor.")