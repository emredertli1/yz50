import torch
import torch.nn as nn
import torch.nn.functional as F

# input.txt: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

torch.manual_seed(1337)

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-2

# 1) Tiny Shakespeare'i oku
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2) Karakter tokenizer
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

# 3) %90 train, %10 val
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 4) Rastgele parçalardan batch
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y

# 5) Bigram modeli (nn.Module)
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

model = BigramLanguageModel(vocab_size)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Eğitim
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters):
    if it % eval_interval == 0:
        l = estimate_loss()
        print(f"adım {it}: train {l['train']:.4f}, val {l['val']:.4f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Val loss'u kaydet (karşılaştırma tabanı)
final = estimate_loss()
print(f"SON: train {final['train']:.4f}, val {final['val']:.4f}")
with open("bigram_val_loss.txt", "w") as f:
    f.write(f"bigram val loss: {final['val']:.4f}\n")