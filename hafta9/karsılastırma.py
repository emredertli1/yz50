import tiktoken

# 1. Aynı anlama gelen Türkçe ve İngilizce paragraflar
tr_metin = "Yapay zeka ve büyük dil modelleri, günümüzde teknolojinin çehresini kökten değiştirmektedir."
en_metin = "Artificial intelligence and large language models are radically transforming the face of technology today."

# 2. GPT-4'ün tokenizer'ı (cl100k_base) yükleniyor
enc = tiktoken.get_encoding("cl100k_base")

tr_token_ids = enc.encode(tr_metin)
en_token_ids = enc.encode(en_metin)

tr_kelime_sayisi = len(tr_metin.split())
en_kelime_sayisi = len(en_metin.split())

tr_oran = len(tr_token_ids) / tr_kelime_sayisi
en_oran = len(en_token_ids) / en_kelime_sayisi

print("=" * 60)
print(f"TÜRKÇE METİN:")
print(f"Kelime Sayısı : {tr_kelime_sayisi}")
print(f"Token Sayısı   : {len(tr_token_ids)}")
print(f"Token / Kelime : {tr_oran:.2f}")
print("-" * 60)
print(f"İNGİLİZCE METİN:")
print(f"Kelime Sayısı : {en_kelime_sayisi}")
print(f"Token Sayısı   : {len(en_token_ids)}")
print(f"Token / Kelime : {en_oran:.2f}")
print("=" * 60)