import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# 1. GPT-2 MİMARİSİ (Sıfırdan Tanımlama)

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
        self.transformer.wte.weight = self.lm_head.weight  # Weight Tying

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained(cls, model_type="gpt2"):
        config = GPTConfig()
        model = cls(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# =============================================================================
# 2. TOP-K AUTOREGRESSIVE SAMPLING DÖNGÜSÜ
# =============================================================================
def generate_text(model, tokenizer, prompt, max_new_tokens=30, top_k=50, temperature=1.0, device="cpu"):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        # 1024 sınırını aşmamak için context penceresini kırp
        idx_cond = tokens if tokens.size(1) <= 1024 else tokens[:, -1024:]

        with torch.no_grad():
            logits = model(idx_cond)
            # Sadece en son üretilen token'ın logits değerleri
            logits = logits[:, -1, :] / temperature

            # Top-k filtreleme
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, idx_next), dim=1)

    return tokenizer.decode(tokens[0].tolist())


# =============================================================================
# 3. ÇALIŞTIRMA VE GÖREV 3 ANALİZİ
# =============================================================================
if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print("=" * 70)
    print(f"Cihaz: {device} | Model ve Tokenizer Yükleniyor...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT.from_pretrained("gpt2").to(device)
    model.eval()
    print("Yükleme Tamamlandı.\n" + "=" * 70)

    # 1. Test: İngilizce Üretim
    torch.manual_seed(42)
    prompt_en = "Hello, I'm a language model,"
    output_en = generate_text(model, tokenizer, prompt=prompt_en, max_new_tokens=25, top_k=50, device=device)
    print("[1] İNGİLİZCE METİN ÜRETİMİ (TOP-K = 50):")
    print(f"Girdi : {prompt_en}")
    print(f"Çıktı : {output_en}\n" + "-" * 70)

    # 2. Test: Türkçe Üretim
    torch.manual_seed(42)
    prompt_tr = "Merhaba, ben bir dil modeliyim,"
    output_tr = generate_text(model, tokenizer, prompt=prompt_tr, max_new_tokens=25, top_k=50, device=device)
    print("[2] TÜRKÇE METİN ÜRETİMİ (TOP-K = 50):")
    print(f"Girdi : {prompt_tr}")
    print(f"Çıktı : {output_tr}\n" + "-" * 70)

    # 3. Test: Tokenizer Karşılaştırması ve Hafta 8 Analizi
    cumle_en = "Hello, I am a language model and I can generate text."
    cumle_tr = "Merhaba, ben bir dil modeliyim ve metin üretebilirim."

    enc_en = tokenizer.encode(cumle_en)
    enc_tr = tokenizer.encode(cumle_tr)

    print("[3] TOKENIZER ANALİZİ VE HAFTA 8 (BPE) BAĞLANTISI:")
    print(f"İngilizce Cümle  : '{cumle_en}'")
    print(f"-> Token Sayısı  : {len(enc_en)}")
    print(f"-> Token Parçaları: {[tokenizer.decode([t]) for t in enc_en]}\n")

    print(f"Türkçe Cümle     : '{cumle_tr}'")
    print(f"-> Token Sayısı  : {len(enc_tr)}")
    print(f"-> Token Parçaları: {[tokenizer.decode([t]) for t in enc_tr]}\n")

    print("TEORİK ANALİZ ÇIKARIMI:")
    print(f"- Sıkıştırma Oranı : İngilizce cümlenin kelime başına token oranı ~{len(enc_en)/11:.2f} iken,")
    print(f"                     Türkçe cümlede bu oran ~{len(enc_tr)/8:.2f} token/kelimedir.")
    print("- Neden Türkçe Kötü?:")
    print("  1. GPT-2'nin WebText eğitim verisinin %95+'i İngilizcedir; Türkçe dilbilgisi kalıplarını görmemiştir.")
    print("  2. Hafta 8'deki BPE mantığı gereği, nadir diller birleştirilemez; kelimeler hece ve byte düzeyine")
    print("     (Byte Fallback) parçalanır. Bu da 1024 token'lık bağlam penceresini (context window) çok hızlı tüketir.")
    print("=" * 70)