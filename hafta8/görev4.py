import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# =========================================
# 1. HİPERPARAMETRELER VE CİHAZ AYARI
# =========================================
batch_size = 64        # Paralel işlenecek veri bloğu sayısı
block_size = 256       # Maksimum bağlam (context) uzunluğu
max_iters = 5000       # Toplam eğitim adımı
eval_interval = 500    # Kaç adımda bir loss değerini ekrana basacağı
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384           # Embedding (Gömme) boyutu
n_head = 6             # Attention (Dikkat) başlığı sayısı
n_layer = 6            # Transformer blok sayısı
dropout = 0.2

torch.manual_seed(1337)

# =========================================
# 2. VERİ YÜKLEME VE HAZIRLIK
# =========================================
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Eşsiz karakterleri (vocabulary) çıkar ve encode/decode haritalarını oluştur
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Veriyi tensor'a çevir ve Train/Val olarak ikiye böl (%90 Train, %10 Val)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# =========================================
# 3. GPT MODEL MİMARİSİ
# =========================================
class Head(nn.Module):
    """ Tek bir Self-Attention başlığı """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Birden fazla Attention başlığının paralel çalışması """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ Basit bir doğrusal katman (MLP) """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Bloğu (Dikkat + İleri Besleme) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =========================================
# 4. MODELİ BAŞLATMA VE EĞİTİM (TRAINING)
# =========================================
model = GPTLanguageModel()
model = model.to(device)

print("="*60)
print(f"Cihaz          : {device.upper()}")
print(f"Vocab Boyutu   : {vocab_size} (Türkçe karakter kümesi)")
print(f"Toplam Parametre: {sum(p.numel() for p in model.parameters()):,}")
print("="*60)
print(f"Model eğitimi başlıyor ({max_iters} adım)...\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_time = time.time()

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(f"Adım {iter:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Süre: {elapsed:.1f}s")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nEğitim Tamamlandı! \n")

# =========================================
# 5. EDEBİ METİN ÜRETİMİ (GENERATION)
# =========================================
print("="*60)
print("MODELİN YAZDIĞI YENİ HİKAYE:")
print("="*60)

# Üretimi başlatmak için ilk bağlam (context)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Modeli değerlendirme moduna al ve 1000 karakterlik çıktı üret
model.eval()
generated_chars = model.generate(context, max_new_tokens=1000)[0].tolist()
print(decode(generated_chars))