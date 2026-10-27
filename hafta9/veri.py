import urllib.request
import urllib.parse
import json

# Wikipedia'nın bot engeline takılmamak için sahte bir tarayıcı kimliği oluşturuyoruz
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/114.0.0.0 Safari/537.36'}

basliklar = [
    "Türkiye", "Osmanlı_İmparatorluğu", "Mustafa_Kemal_Atatürk", "İstanbul", 
    "Ankara", "I._Dünya_Savaşı", "İkinci_Dünya_Savaşı", "Avrupa_Birliği",
    "Türk_dili", "Yapay_zekâ", "Makine_öğrenimi", "Matematik", "Fizik",
    "Roma_İmparatorluğu", "Sovyetler_Birliği", "Soğuk_Savaş", "Kuantum_mekaniği"
]

toplanan_metin = ""

for baslik in basliklar:
    print(f"{baslik} çekiliyor...")
    try:
        url = f"https://tr.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={urllib.parse.quote(baslik)}&format=json"
        
        # İsteğe tarayıcı kimliğimizi (headers) ekliyoruz
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=5)
        
        veri = json.loads(response.read().decode('utf-8'))
        sayfalar = veri['query']['pages']
        for sayfa_id in sayfalar:
            toplanan_metin += sayfalar[sayfa_id].get('extract', '') + "\n\n"
    except Exception as e:
        print(f"Hata: {baslik} atlandı. ({e})")

# input.txt dosyasına yazma
with open("input.txt", "w", encoding="utf-8") as f:
    f.write(toplanan_metin)

print(f"\nİşlem bitti! Dosya başarıyla oluşturuldu.")
print(f"Toplam Karakter Sayısı: {len(toplanan_metin)}")
print(f"Dosya Boyutu: {len(toplanan_metin.encode('utf-8')) / 1024:.0f} KB")