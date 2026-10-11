import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- VERİ VE HİPERPARAMETRELER ---
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "input.txt")
text = open("input.txt", "r", encoding="utf-8").read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

batch_size, block_size = 32, 8
max_iters, eval_iters = 3000, 150
learning_rate = 1e-3
n_embd, n_head = 32, 4
device = "cuda" if torch.cuda.is_available() else "cpu"

log_every = 500             # kaç adımda bir loss yazdırılsın
shape_every = 1500          # kaç adımda bir ÖZET boyut akışı yazdırılsın
detail_stage = "layernorm"  # ayrıntılı boyut akışı sadece bu aşamada (eğitim öncesi) gösterilir
gen_tokens = 400            # sonda üretilecek karakter sayısı

# --- YARDIMCILAR ---
SHAPE = 0  # 0: kapalı | 1: özet boyut akışı | 2: ayrıntılı boyut akışı

def step(name, a, b, lvl=2):
    if SHAPE >= lvl:
        s = lambda x: x if isinstance(x, str) else str(tuple(x.shape))
        print(f"    {name:<46} {s(a):<18} -> {s(b)}")

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i: i + block_size] for i in ix])
    y = torch.stack([d[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

def shape_forward(model, title, level=1):
    global SHAPE
    X, Y = get_batch("val")
    print(f"\n  ===== {title} =====")
    SHAPE = level
    model.eval()
    with torch.no_grad():
        model(X, Y)
    model.train()
    SHAPE = 0

@torch.no_grad()

def estimate_loss(model, split):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(split)
        losses[k] = model(X, Y)[1].item()
    model.train()
    return losses.mean().item()

# --- MODÜLLER ---

class Head(nn.Module):
    def __init__(self, head_size, idx=0):
        super().__init__()
        self.idx = idx  # sadece 0. head'in boyutları yazdırılır
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
        if self.idx == 0:
            step(f"Head0 query: x @ Wq{tuple(self.query.weight.T.shape)}", x, q)
            step(f"Head0 key:   x @ Wk{tuple(self.key.weight.T.shape)}", x, k)
            step(f"Head0 value: x @ Wv{tuple(self.value.weight.T.shape)}", x, v)
            step("Head0 wei = softmax(q @ k^T / sqrt(d))", f"{tuple(q.shape)}x{tuple(k.transpose(-2, -1).shape)}", wei)
            step("Head0 out = wei @ v", f"{tuple(wei.shape)}x{tuple(v.shape)}", out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, with_projection=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, idx=i) for i in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) if with_projection else None
    def forward(self, x):
        outs = [h(x) for h in self.heads]
        out = torch.cat(outs, dim=-1)
        step(f"concat {len(outs)} head x {tuple(outs[0].shape)}", f"{len(outs)} x {tuple(outs[0].shape)}", out)
        if self.proj is not None:
            y = self.proj(out)
            step(f"Projection: out @ Wproj{tuple(self.proj.weight.T.shape)}", out, y)
            out = y
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, expand_4x=False):
        super().__init__()
        self.lin1 = nn.Linear(n_embd, 4 * n_embd if expand_4x else n_embd)
        self.lin2 = nn.Linear(4 * n_embd, n_embd) if expand_4x else None
    def forward(self, x):
        h = self.lin1(x)
        step(f"FFN Linear1: x @ W1{tuple(self.lin1.weight.T.shape)}", x, h)
        r = F.relu(h)
        step("FFN ReLU (negatifler -> 0)", h, r)
        if self.lin2 is None:
            return r
        out = self.lin2(r)
        step(f"FFN Linear2: h @ W2{tuple(self.lin2.weight.T.shape)}", r, out)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head, use_residual=False, use_layernorm=False):
        super().__init__()
        self.use_residual, self.use_layernorm = use_residual, use_layernorm
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, with_projection=use_residual)
        self.ffwd = FeedForward(n_embd, expand_4x=use_residual)
        if use_layernorm:
            self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        if not self.use_residual:
            return self.ffwd(self.sa(x))
        n1 = self.ln1(x) if self.use_layernorm else x
        if self.use_layernorm:
            step("ln1(x)", x, n1)
        a = self.sa(n1)
        x2 = x + a
        step("x + attention (residual)", f"{tuple(x.shape)}+{tuple(a.shape)}", x2)
        n2 = self.ln2(x2) if self.use_layernorm else x2
        if self.use_layernorm:
            step("ln2(x)", x2, n2)
        f = self.ffwd(n2)
        x3 = x2 + f
        step("x + ffwd (residual)", f"{tuple(x2.shape)}+{tuple(f.shape)}", x3)
        return x3

# --- AŞAMALI DİL MODELİ ---

class StageLanguageModel(nn.Module):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if stage != "bigram":
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        hs = n_embd // n_head
        if stage == "single_head":
            self.sa = Head(n_embd)
        elif stage == "multi_head":
            self.sa = MultiHeadAttention(n_head, hs)
        elif stage == "feedforward":
            self.sa = MultiHeadAttention(n_head, hs)
            self.ffwd = FeedForward(n_embd)
        elif stage == "residual":
            self.block = Block(n_embd, n_head, use_residual=True)
        elif stage == "layernorm":
            self.block = Block(n_embd, n_head, use_residual=True, use_layernorm=True)
            self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        step(f"Token embedding tablosu{tuple(self.token_embedding_table.weight.shape)}", idx, x, lvl=1)
        if self.stage != "bigram":
            pos = self.position_embedding_table(torch.arange(T, device=device))
            x2 = x + pos
            step(f"+ Position emb{tuple(pos.shape)} (broadcast)", x, x2, lvl=1)
            x = x2
        xin = x
        if self.stage in ("single_head", "multi_head"):
            x = self.sa(x)
            step("Attention bloğu (toplam)", xin, x, lvl=1)
        elif self.stage == "feedforward":
            x = self.sa(x)
            step("Attention bloğu (toplam)", xin, x, lvl=1)
            x = self.ffwd(x)
            step("FeedForward bloğu (toplam)", xin, x, lvl=1)
        elif self.stage in ("residual", "layernorm"):
            x = self.block(x)
            step("Transformer bloğu (toplam)", xin, x, lvl=1)
            if self.stage == "layernorm":
                xn = self.ln_f(x)
                step("Son LayerNorm (ln_f)", x, xn, lvl=1)
                x = xn
        logits = self.lm_head(x)
        step(f"lm_head: x @ Wlm{tuple(self.lm_head.weight.T.shape)}", x, logits, lvl=1)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
            step("Cross-entropy loss", f"{tuple(logits.view(B * T, -1).shape)} vs {tuple(targets.view(B * T).shape)}",
                 "skaler", lvl=1)
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, show_shapes_steps=1):
        global SHAPE
        self.eval()
        for i in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            SHAPE = 1 if i < show_shapes_steps else 0
            if SHAPE:
                print(f"\n  [üretim adımı {i + 1}] bağlam şekli: {tuple(idx_cond.shape)}")
            logits, _ = self(idx_cond)
            SHAPE = 0
            probs = F.softmax(logits[:, -1, :], dim=-1)          # sadece son konum
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        self.train()
        return idx

# --- TÜM AŞAMALARI EĞİT VE DEĞERLENDİR ---
stages = [("1. Bigram", "bigram"), ("2. Tek Head Attention", "single_head"), ("3. Multi-Head Attention", "multi_head"),
          ("4. + Feedforward", "feedforward"), ("5. + Residual Connections", "residual"), ("6. + LayerNorm (Tam Blok)", "layernorm")]
val_loss_results, param_counts, final_model = {}, {}, None

print(f"Cihaz: {device} | vocab_size: {vocab_size} | (B, T, C) = ({batch_size}, {block_size}, {n_embd}) | n_head: {n_head}")

for display_name, stage_key in stages:
    print("\n" + "#" * 90 + f"\n# AŞAMA: {display_name}\n" + "#" * 90)
    torch.manual_seed(1337)
    model = StageLanguageModel(stage_key).to(device)
    param_counts[display_name] = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre sayısı: {param_counts[display_name]:,}")

    shape_forward(model, "MATRİS BOYUTU AKIŞI (eğitim öncesi)", level=2 if stage_key == detail_stage else 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"\n  Eğitim başlıyor ({max_iters} adım)...")
    for it in range(max_iters):
        if it % log_every == 0 or it == max_iters - 1:
            print(f"  adım {it:5d} | train loss {estimate_loss(model, 'train'):.4f} | val loss {estimate_loss(model, 'val'):.4f}")
        if it > 0 and it % shape_every == 0:
            shape_forward(model, f"adım {it}: özet boyut akışı", level=1)
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    val_loss_results[display_name] = estimate_loss(model, "val")
    print(f"--> {display_name} Tamamlandı | Val Loss: {val_loss_results[display_name]:.4f}")
    if stage_key == "layernorm":
        final_model = model

# --- ÖZET TABLO ---
print("\n" + "=" * 90)
print(f"{'Mimari Aşama':<28} | {'Parametre':<10} | {'Val Loss':<10} | {'Baz.göre':<10} | Önceki aşamaya göre")
print("=" * 90)
base, prev = val_loss_results["1. Bigram"], None
for name, v in val_loss_results.items():
    b = "Referans" if prev is None else f"{base - v:+.4f}"
    p = "-" if prev is None else f"{prev - v:+.4f}"
    print(f"{name:<28} | {param_counts[name]:<10,} | {v:<10.4f} | {b:<10} | {p}")
    prev = v
print("=" * 90)
print("Not: Pozitif fark = iyileşme. 4->5 geçişinde residual + projection + 4x FFN birlikte eklendiği için etki ayrıştırılamaz.")

# --- TAM BLOK MODELİNDEN ÖRNEK YAZI ---
print("\n" + "#" * 90 + "\n# TAM BLOK (LayerNorm) MODELİNDEN ÖRNEK YAZI ÜRETİMİ\n" + "#" * 90)
print(f"Model her adımda son {block_size} karaktere bakıp bir sonraki karakteri tahmin eder.")
torch.manual_seed(42)
for prompt in ["\n", "ROMEO:"]:
    print(f"\n--- Başlangıç: {prompt!r} ---")
    start = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = final_model.generate(start, gen_tokens, show_shapes_steps=1 if prompt == "\n" else 0)
    print("\n" + "-" * 60 + "\n" + decode(out[0].tolist()) + "\n" + "-" * 60)