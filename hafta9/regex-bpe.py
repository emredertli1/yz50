import os
import sys
from typing import Dict, List, Tuple

# GPT-4 regex deseni için üçüncü parti 'regex' kütüphanesi zorunludur.
try:
    import regex as re
except ImportError:
    print("HATA: 'regex' modülü bulunamadı. Lütfen terminalde şunu çalıştırın:")
    print("pip install regex")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. input.txt Dosyasını Yükleme
# ---------------------------------------------------------------------------
# Dosyayı önce bulunulan dizinde, ardından script dizininde arar
dosya_yolu = "input.txt"
if not os.path.exists(dosya_yolu):
    alternatif = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")
    if os.path.exists(alternatif):
        dosya_yolu = alternatif

assert os.path.exists(dosya_yolu), f"'{dosya_yolu}' bulunamadı. Dosyanın aynı klasörde olduğundan emin olun."

with open(dosya_yolu, "r", encoding="utf-8") as f:
    text = f.read()

print("=" * 80)
print(f"Veri Seti: '{dosya_yolu}' | Karakter Sayısı: {len(text):,} | Ham Byte: {len(text.encode('utf-8')):,}")
print("=" * 80)

# ---------------------------------------------------------------------------
# 2. GPT-4 Regex Deseni
# ---------------------------------------------------------------------------
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
gpt4_regex = re.compile(GPT4_SPLIT_PATTERN)

NUM_MERGES = 100

# ---------------------------------------------------------------------------
# 3. Model 1: Düz (Raw) BPE Eğitimi
# ---------------------------------------------------------------------------
print(f"\n[1/2] Düz (Raw) BPE eğitiliyor ({NUM_MERGES} merge)...")
raw_tokens = list(text.encode("utf-8"))
raw_vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
raw_merges: Dict[Tuple[int, int], int] = {}

for i in range(NUM_MERGES):
    counts = {}
    for pair in zip(raw_tokens, raw_tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    if not counts:
        break
    best_pair = max(counts, key=counts.get)
    new_id = 256 + i
    
    # Birleştir
    new_tokens = []
    j = 0
    n = len(raw_tokens)
    p0, p1 = best_pair
    while j < n:
        if j < n - 1 and raw_tokens[j] == p0 and raw_tokens[j + 1] == p1:
            new_tokens.append(new_id)
            j += 2
        else:
            new_tokens.append(raw_tokens[j])
            j += 1
    raw_tokens = new_tokens
    
    raw_merges[best_pair] = new_id
    raw_vocab[new_id] = raw_vocab[best_pair[0]] + raw_vocab[best_pair[1]]

# ---------------------------------------------------------------------------
# 4. Model 2: GPT-4 Regex Ön-Bölümlemeli BPE Eğitimi
# ---------------------------------------------------------------------------
print(f"[2/2] GPT-4 Regex Korumalı BPE eğitiliyor ({NUM_MERGES} merge)...")
text_chunks = gpt4_regex.findall(text)
chunk_tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]

regex_vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
regex_merges: Dict[Tuple[int, int], int] = {}

for i in range(NUM_MERGES):
    # İstatistikleri parçaların sınırlarını ihlal etmeden topla
    counts = {}
    for tokens in chunk_tokens:
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    if not counts:
        break
    best_pair = max(counts, key=counts.get)
    new_id = 256 + i
    
    # Her parçanın kendi içinde merge kuralını işlet
    p0, p1 = best_pair
    new_chunk_tokens = []
    for tokens in chunk_tokens:
        new_ids = []
        j = 0
        n = len(tokens)
        while j < n:
            if j < n - 1 and tokens[j] == p0 and tokens[j + 1] == p1:
                new_ids.append(new_id)
                j += 2
            else:
                new_ids.append(tokens[j])
                j += 1
        new_chunk_tokens.append(new_ids)
    chunk_tokens = new_chunk_tokens
    
    regex_merges[best_pair] = new_id
    regex_vocab[new_id] = regex_vocab[best_pair[0]] + regex_vocab[best_pair[1]]

# ---------------------------------------------------------------------------
# 5. İlk 20 Birleştirmenin Karşılaştırmalı Tablosu
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"{'SIRA':<5} | {'DÜZ BPE (RAW)':<34} | {'GPT-4 REGEX BPE':<34}")
print("=" * 80)

raw_items = list(raw_merges.items())[:20]
regex_items = list(regex_merges.items())[:20]

for idx in range(20):
    # Düz BPE
    r_id = raw_items[idx][1]
    r_bytes = raw_vocab[r_id]
    r_str = repr(r_bytes.decode("utf-8", errors="replace"))
    
    # Regex BPE
    rgx_id = regex_items[idx][1]
    rgx_bytes = regex_vocab[rgx_id]
    rgx_str = repr(rgx_bytes.decode("utf-8", errors="replace"))
    
    raw_col = f"ID {r_id}: {r_str}"
    rgx_col = f"ID {rgx_id}: {rgx_str}"
    print(f"{idx+1:<5} | {raw_col:<34} | {rgx_col:<34}")

print("=" * 80)

# ---------------------------------------------------------------------------
# 6. Kesme İşaretli Kelimelerin GPT-4 Regex Deseniyle Test Edilmesi
# ---------------------------------------------------------------------------
test_kelimeleri = [
    "Ankara'dan",
    "İstanbul'da",
    "Ahmet'in",
    "İzmir'e",
    "Türkiye'de",
    "don't",
    "I'd",
    "they're"
]

print("\nGPT-4 REGEX DESENİNİN KELİME PARÇALAMA TESTİ:")
print("-" * 65)
for kelime in test_kelimeleri:
    parcalar = gpt4_regex.findall(kelime)
    print(f"{kelime:<16} -> {parcalar}")
print("-" * 65)