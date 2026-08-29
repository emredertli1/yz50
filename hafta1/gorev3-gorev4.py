import matplotlib.pyplot as plt
# GÖREV 3  Bir Loss Fonksiyonu (MSE)
def mean_squared_error(y_true, y_pred):
    """
    Kayıp (Loss) Fonksiyonu: Ortalama Kare Hata (MSE)
    Formül: (1/n) * sum((y_pred - y_true)^2)
    """
    total_loss = sum((yp - yt) ** 2 for yt, yp in zip(y_true, y_pred))
    return total_loss / len(y_true)
    #GÖREV 3 TEST
test_gercek = [10.0]
test_tahmin = [8.0]
print("Görev 3 Test Loss Değeri:", mean_squared_error(test_gercek, test_tahmin))

# GÖREV 4 Manuel Parametre Değişimi ve Loss Eğrisi Çizimi

x = 2.0
y_true = [10.0]

w_degerleri = []
loss_degerleri = []

# w parametresini -2.0 ile 12.0 arasında 0.2 adımlarla manuel değiştirelim
w = -2.0
while w <= 12.0:
    # 1. Forward Pass (Tek nöron, bias=0 varsayımı)
    y_pred = [w * x]
    
    # 2. Loss hesapla
    loss = mean_squared_error(y_true, y_pred)
    
    # Değerleri listeye kaydet
    w_degerleri.append(w)
    loss_degerleri.append(loss)
    
    w += 0.2

#  Loss Eğrisi
plt.figure(figsize=(7, 4))
plt.plot(w_degerleri, loss_degerleri, color='blue', linewidth=2, label='Loss Eğrisi (MSE)')
plt.axvline(x=5.0, color='red', linestyle='--', label='Minimum Hata (w = 5.0)')

plt.title("Parametreye (W) Göre Loss Değişimi")
plt.xlabel("Ağırlık Değeri (w)")
plt.ylabel("Loss (Hata)")
plt.grid(True)
plt.legend()
plt.show()