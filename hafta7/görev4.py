import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
n_embd = 32

# Veri Hazırlığı
with open("input.txt", "r", encoding="utf-8") as f:
  text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
  d = train_data if split == "train" else val_data
  ix = torch.randint(len(d) - block_size, (batch_size,))
  x = torch.stack([d[i : i + block_size] for i in ix])
  y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
  return x, y


# 1. Attention Head (wei matrisini dışarı döndürecek şekilde güncellendi)
class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    k, q, v = self.key(x), self.query(x), self.value(x)
    wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    out = wei @ v
    return out, wei


# 2. Dil Modeli (wei matrisini forward üzerinden iletir)
class BigramWithHead(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.sa_head = Head(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    x = self.token_embedding_table(idx) + self.position_embedding_table(
        torch.arange(T, device=idx.device)
    )
    x, wei = self.sa_head(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
    return logits, loss, wei

  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]  # son block_size harfe kırp
      logits, _, _ = self(idx_cond)
      probs = F.softmax(logits[:, -1, :], dim=-1)
      idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    return idx


model = BigramWithHead()


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      _, loss, _ = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean().item()
  model.train()
  return out


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Eğitim başlıyor...")
for it in range(max_iters):
  if it % eval_interval == 0:
    l = estimate_loss()
    print(f"adım {it}: train {l['train']:.4f}, val {l['val']:.4f}")
  xb, yb = get_batch("train")
  _, loss, _ = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

final = estimate_loss()
print(f"\nhead'li model val loss: {final['val']:.4f}")

if os.path.exists("bigram_val_loss.txt"):
  base = float(open("bigram_val_loss.txt").read().split(":")[1])
  print(f"bigram tabanı val loss: {base:.4f}")
  print(f"fark: {base - final['val']:.4f}")

# Metin Üretimi
context = torch.zeros((1, 1), dtype=torch.long)
print("\nÜretilen Metin:")
print(decode(model.generate(context, 200)[0].tolist()))
print("-" * 50)

# ==========================================
# EK GÖREV (a): Isı Haritası Çizimi ve Analiz
# ==========================================
model.eval()
sample_text = "thou art"  # 8 karakterlik bağlam
idx_sample = torch.tensor([encode(sample_text)], dtype=torch.long)

with torch.no_grad():
  _, _, sample_wei = model(idx_sample)

matrix = sample_wei[0].cpu().numpy()

# Terminal Çıktısı (Okuma)
print(f"\n'{sample_text}' Cümlesi İçin Attention Analizi:")
for i, ch_q in enumerate(sample_text):
  print(f"Sıra {i} ('{ch_q}'):")
  for j, ch_k in enumerate(sample_text[: i + 1]):
    print(f"   -> '{ch_k}' (indeks {j}): %{matrix[i, j] * 100:.1f}")

# Matplotlib ile Isı Haritası Kaydetme
plt.figure(figsize=(7, 6))
plt.imshow(matrix, cmap="Blues")
chars_labels = [f"'{c}'" if c != " " else "'space'" for c in sample_text]
plt.xticks(range(len(sample_text)), chars_labels, fontsize=10)
plt.yticks(range(len(sample_text)), chars_labels, fontsize=10)
plt.xlabel("Key (Bakılan Karakter)", fontsize=11)
plt.ylabel("Query (Arayan Karakter)", fontsize=11)
plt.title("Eğitilmiş Modelin Dikkat Isı Haritası", fontsize=12)
plt.colorbar(label="Dikkat Ağırlığı (Softmax Çıktısı)")

for i in range(len(sample_text)):
  for j in range(len(sample_text)):
    val = matrix[i, j]
    text_color = "white" if val > 0.45 else "black"
    plt.text(
        j,
        i,
        f"{val:.2f}",
        ha="center",
        va="center",
        color=text_color,
        fontsize=9,
    )

plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150)
print("\nIsı haritası 'attention_heatmap.png' olarak başarıyla kaydedildi.")