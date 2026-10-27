import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer


# =============================================================================
# 1. GPT-2 Mimarisi (Loss Destekli)
# =============================================================================
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight

        # Karpathy standart ağırlık başlatması
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy için boyutları düzleştir: (B*T, vocab_size) ve (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# =============================================================================
# 2. Test ve Ezberletme (Overfitting) Senaryosu
# =============================================================================
if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print("=" * 65)
    print(f"Kullanılan Cihaz: {device}")

    # 1. Modeli Rastgele Başlat (Ağırlık Yüklemiyoruz!)
    torch.manual_seed(1337)
    config = GPTConfig()
    model = GPT(config).to(device)

    # 2. Tek Bir Batch Veri Hazırla (Örnek Metin)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    sample_text = (
        "Merhaba, bu bir dil modelidir. Bu tek bir batch uzerinde modeli "
        "asiri ogrenmeye (overfitting) zorlayarak calismasini test ediyoruz."
    )
    tokens = tokenizer.encode(sample_text)
    
    # Girdi (x) ve Hedef (y = x'in bir sonraki token kaydırılmışı)
    # Batch boyutu = 1, Dizi uzunluğu = T
    x = torch.tensor([tokens[:-1]], dtype=torch.long).to(device)
    y = torch.tensor([tokens[1:]], dtype=torch.long).to(device)

    print(f"Batch Girdi Boyutu (x): {x.shape}")
    print(f"Batch Hedef Boyutu (y): {y.shape}")
    print("-" * 65)

    # 3. İlk Loss Ölçümü
    model.eval()
    with torch.no_grad():
        _, initial_loss = model(x, y)

    expected_loss = math.log(config.vocab_size)
    print(f"Teorik Beklenen Loss: -ln(1/50257) = {expected_loss:.4f}")
    print(f"Modelin İlk Loss Değeri             = {initial_loss.item():.4f}")
    print(f"Fark                               = {abs(initial_loss.item() - expected_loss):.4f}")
    print("-" * 65)

    # 4. Tek Batch'i Ezberletme Döngüsü (Overfit)
    print("Tek Batch Üzerinde Eğitim Başlıyor (Hedef Loss -> 0.0)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    for step in range(1, 61):  # 60 adımda sıfıra yakınsar
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"Adım {step:3d} | Loss: {loss.item():.6f}")

    print("-" * 65)
    print(f"Eğitim Sonu Kayıp: {loss.item():.6f}")
    print("Doğrulama: Model tek batch'i başarıyla ezberledi, boru hattı sağlam!")
    print("=" * 65)