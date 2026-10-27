import os
from typing import Dict, Tuple, List

# ---------------------------------------------------------------------------
# 1. input.txt Dosyasını Oku
# ---------------------------------------------------------------------------
file_path = "input.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"'{file_path}' bulunamadı. Lütfen geçen haftaki Türkçe veri setinizi bu script ile aynı dizine koyun."
    )

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Büyük dosyalarda saf Python döngüsünün hızlı tamamlanması için metin boyutu
# İsteğe bağlı olarak ilk 100.000 karakterle sınırlandırılabilir:
# text = text[:100000]

print(f"'{file_path}' başarıyla yüklendi. Karakter sayısı: {len(text):,}")

# ---------------------------------------------------------------------------
# 2. Temel Fonksiyonlar: get_stats ve merge
# ---------------------------------------------------------------------------

def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Token dizisindeki ardışık ikililerin (pair) frekansını sayar."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Dizideki hedeflenen 'pair' ikililerini yeni 'idx' tokenı ile birleştirir."""
    new_ids = []
    i = 0
    n = len(ids)
    p0, p1 = pair
    
    while i < n:
        if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# ---------------------------------------------------------------------------
# 3. BPE Eğitimi
# ---------------------------------------------------------------------------

raw_bytes = text.encode("utf-8")
tokens = list(raw_bytes)

# Hedef sözlük boyutu: 256 taban byte + belirlenen sayıda birleştirme
vocab_size = 356  # 100 birleştirme (merge)
num_merges = vocab_size - 256

# Başlangıç sözlüğü: Her byte değeri kendi tekil byte nesnesiyle başlar
vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
merges: Dict[Tuple[int, int], int] = {}

print(f"Başlangıç byte sayısı : {len(tokens):,}")
print(f"Yapılacak birleştirme : {num_merges}\nEğitim başladı...")

for i in range(num_merges):
    stats = get_stats(tokens)
    if not stats:
        break
    
    best_pair = max(stats, key=stats.get)
    new_id = 256 + i
    
    tokens = merge(tokens, best_pair, new_id)
    
    merges[best_pair] = new_id
    vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
    
    if (i + 1) % 20 == 0 or (i + 1) == num_merges:
        print(f"[{i + 1:3d}/{num_merges}] Tamamlandı | Kalan token sayısı: {len(tokens):,}")

print("\nEğitim bitti!\n")

# ---------------------------------------------------------------------------
# 4. İlk 20 Birleştirmenin İncelenmesi
# ---------------------------------------------------------------------------

print("=" * 70)
print(f"{'Sıra':<5} | {'Token ID':<9} | {'Bileşenler':<14} | {'Metin Karşılığı':<25}")
print("=" * 70)

for i, (pair, new_id) in enumerate(list(merges.items())[:20]):
    # Hatalı/yarım kalan byte parçalarını güvenli decode etmek için errors='replace'
    token_str = vocab[new_id].decode("utf-8", errors="replace")
    
    # Görünmeyen boşluk ve satır başlarını net göstermek için repr()
    display_str = repr(token_str)
    print(f"{i + 1:<5} | {new_id:<9} | {str(pair):<14} | {display_str:<25}")

print("=" * 70)

# ---------------------------------------------------------------------------
# 5. Sıkıştırma Oranı Hesabı
# ---------------------------------------------------------------------------

byte_count = len(raw_bytes)
token_count = len(tokens)
compression_ratio = byte_count / token_count

print("\nPERFORMANS RAPORU:")
print(f"Orijinal Metin Boyutu (Byte) : {byte_count:,}")
print(f"BPE Sonrası Boyut (Token)    : {token_count:,}")
print(f"Sıkıştırma Oranı (Byte/Token): {compression_ratio:.2f}X")
print(f"Tasarruf Oranı               : %{(1 - (token_count / byte_count)) * 100:.1f}")