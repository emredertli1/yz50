import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# -----------------------------------------------------------------------------
# 1. Konfigürasyon
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024   # wpe boyutu (maksimum context)
    vocab_size: int = 50257  # wte boyutu (sözlük)
    n_layer: int = 12        # Katman sayısı
    n_head: int = 12         # Dikkat kafası sayısı
    n_embd: int = 768        # Embedding boyutu (d_model)


# -----------------------------------------------------------------------------
# 2. Causal Self-Attention Katmanı
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Q, K, V tek matriste hesaplanır: (768 -> 2304)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal maske (gelecekteki token'ları engellemek için)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x):
        B, T, C = x.size()

        # (B, T, 2304) -> 3 adet (B, T, 768)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 12 kafa ve 64 boyuta bölme: (B, nh, T, hs)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# -----------------------------------------------------------------------------
# 3. MLP (Feed-Forward Network) Katmanı
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)      # 768 -> 3072
        self.gelu = nn.GELU(approximate="tanh")                       # GPT-2 Tanh GELU
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)    # 3072 -> 768

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


# -----------------------------------------------------------------------------
# 4. Transformer Bloğu
# -----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN ve Residual akış
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -----------------------------------------------------------------------------
# 5. GPT Modeli
# -----------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying (wte ve lm_head bellek paylaşımı)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Girdi {T} token, maksimum {self.config.block_size} desteklenir."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        """Hugging Face ağırlıklarını indirip transpoze ederek modele yükler."""
        config = GPTConfig()
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        # Hugging Face GPT-2'yi çek
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]

        # Conv1D formatından nn.Linear'a geçmek için transpoze edilecek katmanlar
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight"
        ]

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Ters boyut kontrolü ve .t() işlemi
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# -----------------------------------------------------------------------------
# 6. Test ve Doğrulama
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"Kullanılan Donanım: {device}")
    print("Modeller yükleniyor...")

    # Kendi modelimizi yükle
    my_gpt = GPT.from_pretrained("gpt2").to(device)
    my_gpt.eval()

    # Orijinal Hugging Face modelini yükle
    hf_gpt = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    hf_gpt.eval()

    # Test girdisi hazırla
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_text = "Hello, I'm a language model,"
    tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # İki modelden de tahmin üret
    with torch.no_grad():
        my_logits = my_gpt(tokens)
        hf_logits = hf_gpt(tokens).logits

    # Karşılaştırma
    max_diff = (my_logits - hf_logits).abs().max().item()
    is_match = torch.allclose(my_logits, hf_logits, atol=1e-4)

    print("=" * 60)
    print(f"Girdi                  : {input_text}")
    print(f"Logits Boyutu          : {my_logits.shape}")
    print(f"Maksimum Mutlak Fark   : {max_diff:.8e}")
    print(f"Eşleşme Durumu (Match) : {'BAŞARILI (Birebir Aynı)' if is_match else 'HATA'}")
    print("=" * 60)