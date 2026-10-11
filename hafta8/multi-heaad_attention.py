import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F


# Veri Seti (Tiny Shakespeare) ve Hazırlık

if not os.path.exists("input.txt"):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "input.txt")

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hiperparametreler 
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 32


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Modüller: Single Head ve Multi-Head

class Head(nn.Module):
    """Tek bir self-attention başlığı"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    """Paralel çalışan ve çıktıları birleştiren birden fazla başlık"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

# 2. Dil Modeli Şablonu

class LanguageModel(nn.Module):

    def __init__(self, attention_type="single"):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        if attention_type == "single":
            # 1 adet 32 boyutlu başlık
            self.sa = Head(head_size=n_embd)
        elif attention_type == "multi":
            # 4 adet 8 boyutlu paralel başlık (4 x 8 = 32 boyut)
            self.sa = MultiHeadAttention(num_heads=4, head_size=n_embd // 4)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# 3. İki Modeli Eğitme ve Loss Değerlerini Kaydetme

def train_model(model_type):
    torch.manual_seed(1337)
    model = LanguageModel(attention_type=model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = {}
    for iter in range(max_iters + 1):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            history[iter] = losses

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return history


print("Single Head modeli eğitiliyor...")
single_history = train_model("single")

print("Multi-Head Attention modeli eğitiliyor...")
multi_history = train_model("multi")

# 4. Yan Yana Karşılaştırma Tablosu

print("\n" + "=" * 65)
print(f"{'Adım (Step)':<12} | {'Single Head Val Loss':<22} | {'Multi-Head Val Loss':<20}")
print("=" * 65)

for step in sorted(single_history.keys()):
    s_val = single_history[step]["val"]
    m_val = multi_history[step]["val"]
    diff = s_val - m_val
    print(f"{step:<12} | {s_val:<22.4f} | {m_val:<20.4f} (Fark: {diff:+.4f})")

print("=" * 65)