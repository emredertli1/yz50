from typing import Dict, Tuple, List

# ---------------------------------------------------------------------------
# 1. Küçük Bir BPE Modeli Eğitelim (Örnek Sözlük ve Kural Üretimi)
# ---------------------------------------------------------------------------
corpus = """
Bugün hava çok güzel ☀️, öğleden sonra sahilde yürüyüş yapıp kahve içeceğim ☕.
Çocuklar parkta neşeyle koşuyor 🏃, kuşlar ağaçların üstünde ötüyor 🌳.
Kütüphaneden yeni kitaplar aldım 📚, akşam evde sakin bir şekilde okuyacağım 🛋️.
Masada bilgisayarı açtım 💻, küçük hataları tek tek düzeltiyorum 🔍.
Pazardan taze elma 🍏, portakal 🍊 ve fıstık aldık 🥜.
""" * 10

def train_mini_bpe(text: str, target_vocab_size: int = 320):
    raw_bytes = text.encode("utf-8")
    tokens = list(raw_bytes)
    num_merges = target_vocab_size - 256
    
    vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
    merges: Dict[Tuple[int, int], int] = {}
    
    def get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_ids(ids, pair, idx):
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

    for i in range(num_merges):
        stats = get_stats(tokens)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        new_id = 256 + i
        tokens = merge_ids(tokens, best_pair, new_id)
        merges[best_pair] = new_id
        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
    return merges, vocab

# Kuralları ve sözlüğü oluşturalım
merges, vocab = train_mini_bpe(corpus, target_vocab_size=320)

# ---------------------------------------------------------------------------
# 2. encode ve decode Fonksiyonları
# ---------------------------------------------------------------------------

def encode(text: str, merges: Dict[Tuple[int, int], int]) -> List[int]:
    """
    Düz metni alır, UTF-8 byte'larına çevirir ve merges kurallarını 
    öğrenilme sırasına (öncelik sırasına) göre işleterek token listesi döndürür.
    """
    # 1. Metni UTF-8 ham byte tamsayılarına çevir
    raw_bytes = text.encode("utf-8")
    ids = list(raw_bytes)
    
    # 2. Metinde birleşebilecek en az 2 token olduğu sürece devam et
    while len(ids) >= 2:
        # Metindeki mevcut tüm ardışık ikilileri bul
        stats = {}
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        
        # Bu ikililerden merges kural tablomuzda olanları filtrele
        # merges içindeki değeri en KÜÇÜK olan ikiliyi seçiyoruz.
        # Neden? Çünkü ID'si küçük olan kural eğitimde DAHA ÖNCE öğrenilmiştir (en yüksek öncelik).
        candidate_pairs = {pair: merges[pair] for pair in stats if pair in merges}
        
        # Eğer metindeki hiçbir ikili kural tablomuzda yoksa işlem tamamlanmıştır
        if not candidate_pairs:
            break
            
        best_pair = min(candidate_pairs, key=candidate_pairs.get)
        new_token_id = merges[best_pair]
        
        # Metinde bu ikiliyi birleştir
        new_ids = []
        i = 0
        n = len(ids)
        p0, p1 = best_pair
        while i < n:
            if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(new_token_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids
        
    return ids

def decode(ids: List[int], vocab: Dict[int, bytes]) -> str:
    """
    Token ID listesini alır, vocab sözlüğünden byte karşılıklarını uç uca ekler
    ve UTF-8 string olarak döner. Hatalı byte dizilimlerinde çökmeyi engeller.
    """
    # Her bir token ID'sinin temsil ettiği byte bloklarını birleştir
    token_bytes = b"".join([vocab[idx] for idx in ids])
    
    # Byte dizisini UTF-8 metne çevir; geçersiz dizilimlerde 'replace' kullan
    text = token_bytes.decode("utf-8", errors="replace")
    return text

# ---------------------------------------------------------------------------
# 3. Test: decode(encode(metin)) == metin (Doğrulama Testi)
# ---------------------------------------------------------------------------

test_cumlesi = "Fıstıkçı Şahap, ılık sütü ve poğaçayı çok sever 🚀🤖🇹🇷!"

token_ids = encode(test_cumlesi, merges)
geri_cozulen_metin = decode(token_ids, vocab)

print("=" * 75)
print(f"Orijinal Metin   : {test_cumlesi}")
print(f"Token Dizisi     : {token_ids}")
print(f"Token Sayısı     : {len(token_ids)} (Orijinal byte sayısı: {len(test_cumlesi.encode('utf-8'))})")
print(f"Decode Çıktısı   : {geri_cozulen_metin}")
print("-" * 75)

# Birebir eşitlik doğrulaması (Lossless Roundtrip)
assert geri_cozulen_metin == test_cumlesi, "HATA: Decode edilen metin orijinali ile uyuşmuyor!"
print("DOĞRULAMA BAŞARILI: decode(encode(metin)) == metin eşitliği sağlandı.")
print("=" * 75)

# ---------------------------------------------------------------------------
# 4. Geçersiz UTF-8 Token Dizisi Deneyi
# ---------------------------------------------------------------------------

print("\nGEÇERSİZ UTF-8 DENEYİ:")

# Türkçe 'ş' karakteri 2 byte tutar: 0xC5 (197) ve 0x9E (158)
# Eğer bir token dizisinde sadece ilk byte (197) yalnız kalırsa:
gecersiz_tokenlar = [197]  # Tek başına duran yarım bir UTF-8 byte'ı

print(f"Yalnız bırakılan yarım byte ID'si : {gecersiz_tokenlar}")

# errors='replace' ile decode edilirse:
guvenli_cikti = decode(gecersiz_tokenlar, vocab)
print(f"decode(gecersiz_tokenlar) Çıktısı : '{guvenli_cikti}' (Unicode: {guvenli_cikti.encode('unicode_escape')})")

# errors='strict' olsaydı ne olurdu?
try:
    hatali_bytes = b"".join([vocab[idx] for idx in gecersiz_tokenlar])
    hatali_bytes.decode("utf-8", errors="strict")
except UnicodeDecodeError as e:
    print(f"errors='strict' Kullanılsaydı Alınacak Hata: {type(e).__name__} -> {e}")