import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel

# 1. Mac M3 için donanım hızlandırmayı ayarla
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}\n")

# 2. Hugging Face'ten GPT-2 (124M) ağırlıklarını yükle
print("GPT-2 (124M) modeli yükleniyor...\n")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

# 3. Ağırlıkların isimlerini, boyutlarını ve toplam parametre sayısını bul
toplam_parametre = 0
print("=" * 70)
print(f"{'Katman Adı':<50} | {'Boyut (Shape)'}")
print("=" * 70)

for isim, parametre in model.named_parameters():
    print(f"{isim:<50} | {tuple(parametre.shape)}")
    toplam_parametre += parametre.numel()

print("=" * 70)
print(f"Toplam Parametre Sayısı: {toplam_parametre:,}")
print("=" * 70)

# 4. Pozisyon Embedding'lerini (wpe) Çek ve Görselleştir
# Modelin 'wpe' katmanı: transformer.wpe.weight (Boyutu: 1024 x 768)
wpe = model.transformer.wpe.weight.detach().cpu().numpy()

plt.figure(figsize=(10, 8))
# cmap="bwr" (Mavi-Beyaz-Kırmızı) ağırlıkların negatif/pozitif dağılımını iyi gösterir
plt.imshow(wpe, cmap="bwr", aspect="auto")
plt.title("GPT-2 Pozisyon Embedding Matrisi (wpe)")
plt.xlabel("Gömme Boyutu (Embedding Dimension - 768)")
plt.ylabel("Token Pozisyonu (Context Length - 1024)")
plt.colorbar(label="Ağırlık Değeri")
plt.tight_layout()

print("\nGrafik çiziliyor... Lütfen açılan penceredeki görseli incele.")
plt.show()