import torch
import torch.nn.functional as F

torch.manual_seed(1337)

# Basit bir tensör boyutu tanımlıyoruz:
# B = 4 (Batch boyutu), T = 8 (Zaman / Bağlam penceresi), C = 2 (Kanal / Özellik boyutu)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# 1. YOL: For Döngüsü ile Geçmişin Ortalamasını Alma
# Her batch ve her zaman adımı için 0'dan t'ye kadar olan dilimi kesip mean(0) alırız.
xbow1 = torch.zeros((B, T, C))
for b in range(B):
  for t in range(T):
    xprev = x[b, : t + 1]  # (t+1, C) boyutunda geçmiş dilimi
    xbow1[b, t] = torch.mean(xprev, 0)

# 2. YOL: torch.tril ve Matris Çarpımı
# Alt üçgen matris oluşturup satır toplamlarına bölerek satır bazında normalize ederiz.
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
# wei: (T, T), x: (B, T, C) -> PyTorch broadcasting ile (B, T, T) @ (B, T, C) yapar
xbow2 = wei @ x

# 3. YOL: Maskeleme ve Softmax
# Gelecekteki adımları -inf yaparak softmax'e sokarız.
# e^(-inf) = 0 olduğu için geleceğin payı 0 olur, geçmiş adımlar normalize edilir.
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x

# DOĞRULAMA (Üç yolun sayısal denkliği)
print("1. Yol ve 2. Yol eşit mi?:", torch.allclose(xbow1, xbow2))
print("1. Yol ve 3. Yol eşit mi?:", torch.allclose(xbow1, xbow3))
print("Maksimum fark (1 vs 2):", (xbow1 - xbow2).abs().max().item())
print("Maksimum fark (1 vs 3):", (xbow1 - xbow3).abs().max().item())