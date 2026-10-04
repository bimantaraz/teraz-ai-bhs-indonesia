import argparse
import json
import math
import os
import sys
import time
import warnings
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

if os.name == 'nt':
    os.system('')

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GRAY = "\033[90m"
    BOLD_CYAN = "\033[1;96m"
    BOLD_GREEN = "\033[1;92m"

USE_COLOR = sys.stdout.isatty()

def c(text: str, color_code: str) -> str:
    return f"{color_code}{text}{Colors.RESET}" if USE_COLOR else text

PRESETS = {
    "balanced": {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.0,
        "repetition_penalty": 1.1,
        "desc": "Keseimbangan kreativitas dan akurasi (default)",
    },
    "creative": {
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "desc": "Kreatif & bervariasi (puisi, cerita, ide)",
    },
    "precise": {
        "temperature": 0.3,
        "top_k": 20,
        "top_p": 0.0,
        "repetition_penalty": 1.15,
        "desc": "Fokus & faktual (pengetahuan, fakta)",
    },
}

SPECIAL_TOKENS = ["<pad>", "<eos>", "<unk>", "<|user|>", "<|assistant|>"]
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2
USER_ID = 3
ASST_ID = 4
NUM_SPECIAL = len(SPECIAL_TOKENS)

def format_number(n: int) -> str:
    return f"{n:,}".replace(",", ".")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def setup_cuda_perf():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

def get_amp_autocast(device_type: str, enabled: bool, dtype: torch.dtype = torch.float16):
    try:
        return torch.amp.autocast(device_type, enabled=enabled, dtype=dtype)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)

def select_amp_dtype(preferred: str) -> Tuple[str, torch.dtype, bool]:
    if preferred == 'fp32':
        return 'fp32', torch.float32, False
    if preferred == 'fp16':
        return 'fp16', torch.float16, True
    if preferred == 'bf16':
        return 'bf16', torch.bfloat16, False

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return 'bf16', torch.bfloat16, False
    if torch.cuda.is_available():
        return 'fp16', torch.float16, True
    return 'fp32', torch.float32, False

class CausalSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) harus habis dibagi n_heads ({n_heads})"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderTransformer(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_seq: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq = max_seq

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq, f"Sequence length {T} melebihi max {self.max_seq}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)

def resolve_checkpoint(model_dir: str, name: str) -> str:
    def latest_checkpoint() -> Optional[str]:
        found = []
        if os.path.isdir(model_dir):
            for f in os.listdir(model_dir):
                if f.startswith("checkpoint_step_") and f.endswith(".pt"):
                    try:
                        found.append((int(f.split("_")[2].split(".")[0]), f))
                    except (IndexError, ValueError):
                        pass
        return max(found)[1] if found else None

    if name == "best":
        if os.path.exists(os.path.join(model_dir, "Teraz-AI.pt")):
            return "Teraz-AI.pt"
        if os.path.exists(os.path.join(model_dir, "best_model.pt")):
            return "best_model.pt"
        fallback = latest_checkpoint()
        if fallback:
            return fallback
        if os.path.isdir(model_dir):
            pts = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
            if pts:
                return pts[0]
        return "Teraz-AI.pt"
    if name == "latest":
        return latest_checkpoint() or "Teraz-AI.pt"
    return name

def load_model(model_dir: str, checkpoint_name: str, device: torch.device):
    from tokenizers import Tokenizer

    checkpoint_name = resolve_checkpoint(model_dir, checkpoint_name)

    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    config_path = os.path.join(model_dir, "model_config.json")
    ckpt_path = os.path.join(model_dir, checkpoint_name)

    for path, label in [(tokenizer_path, "tokenizer.json"),
                        (config_path, "model_config.json"),
                        (ckpt_path, checkpoint_name)]:
        if not os.path.exists(path):
            print(f"Error: '{label}' tidak ditemukan di '{model_dir}'.")
            sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer = Tokenizer.from_file(tokenizer_path)

    model = DecoderTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        max_seq=config["max_seq"],
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    del ckpt, state_dict

    return model, tokenizer, config

class StreamDecoder:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: List[int] = []
        self.printed = ""

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tokenizer.decode(self.ids)
        if text.endswith("\ufffd"):
            return ""
        new = text[len(self.printed):]
        self.printed = text
        return new

    def flush(self) -> str:
        text = self.tokenizer.decode(self.ids)
        new = text[len(self.printed):]
        self.printed = text
        return new

@torch.no_grad()
def generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_seq: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    stream: bool = True,
) -> Tuple[List[int], str]:
    generated_ids: List[int] = []
    decoder = StreamDecoder(tokenizer)

    seen_tokens = {t for t in input_ids[0].tolist() if t >= NUM_SPECIAL}

    finish_reason = "length"

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= max_seq:
            finish_reason = "context"
            break

        with get_amp_autocast('cuda', enabled=(device.type == 'cuda'), dtype=amp_dtype):
            logits = model(input_ids)

        next_logits = logits[:, -1, :].float() / max(temperature, 1e-8)

        next_logits[:, PAD_ID] = float('-inf')
        next_logits[:, UNK_ID] = float('-inf')
        next_logits[:, ASST_ID] = float('-inf')

        if repetition_penalty != 1.0 and seen_tokens:
            idx = torch.tensor(sorted(seen_tokens), device=device)
            vals = next_logits[0, idx]
            vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            next_logits[0, idx] = vals

        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            topk_vals, _ = torch.topk(next_logits, k)
            next_logits[next_logits < topk_vals[:, [-1]]] = float('-inf')

        if top_p > 0.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cum_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_logits[remove_mask] = float('-inf')
            next_logits = torch.full_like(next_logits, float('-inf'))
            next_logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        if token_id == EOS_ID:
            finish_reason = "eos"
            break
        if token_id == USER_ID:
            finish_reason = "eos"
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_ids.append(token_id)
        seen_tokens.add(token_id)

        piece = decoder.push(token_id)
        if stream and piece:
            print(piece, end="", flush=True)

    tail = decoder.flush()
    if stream and tail:
        print(tail, end="", flush=True)

    return generated_ids, finish_reason

def build_context(history_turns: List[List[int]], turn_ids: List[int],
                  max_seq: int, max_new_tokens: int) -> List[int]:
    budget = max(max_seq - max_new_tokens, max_seq // 2)

    turns = list(history_turns)
    total = sum(len(t) for t in turns) + len(turn_ids)
    while turns and total > budget:
        total -= len(turns.pop(0))

    context: List[int] = [tok for t in turns for tok in t] + turn_ids

    if len(context) > budget:
        context = [USER_ID] + context[-(budget - 1):]
    return context

def main():
    parser = argparse.ArgumentParser(
        description="Chat interaktif dengan model Decoder-Only Transformer (dataset 3M)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default=".",
                        help="Folder berisi tokenizer.json, model_config.json, dan checkpoint")
    parser.add_argument("--checkpoint", default="best",
                        help="'best' (best_model.pt / Sekarang Teraz-AI.pt), 'latest' (checkpoint step terbaru), "
                             "atau nama file checkpoint (relatif terhadap --model-dir)")
    parser.add_argument("--preset", choices=["balanced", "creative", "precise"], default="balanced",
                        help="Preset sampling: balanced (default), creative, precise")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Jumlah maksimum token yang di-generate per respons")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature sampling (override preset)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (override preset)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (override preset)")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Penalti token yang sudah muncul (override preset)")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                        help="Dtype inference (auto = bf16 di GPU Ampere+)")
    parser.add_argument("--no-history", action="store_true",
                        help="Jangan pertahankan histori percakapan antar giliran")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed opsional untuk hasil yang bisa direproduksi")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    active_preset = PRESETS.get(args.preset, PRESETS["balanced"])
    temperature = args.temperature if args.temperature is not None else active_preset["temperature"]
    top_k = args.top_k if args.top_k is not None else active_preset["top_k"]
    top_p = args.top_p if args.top_p is not None else active_preset["top_p"]
    repetition_penalty = args.repetition_penalty if args.repetition_penalty is not None else active_preset["repetition_penalty"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        setup_cuda_perf()

    amp_label, amp_dtype, _ = select_amp_dtype(args.dtype)

    print("  Memuat Teraz AI...", end="", flush=True)

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.exists(os.path.join(model_dir, "tokenizer.json")):
        alt_dir = os.path.join(model_dir, "model_output_3M")
        if os.path.exists(os.path.join(alt_dir, "tokenizer.json")):
            model_dir = alt_dir

    model, tokenizer, config = load_model(model_dir, args.checkpoint, device)
    max_seq = config["max_seq"]

    if args.max_new_tokens >= max_seq:
        args.max_new_tokens = max_seq // 2

    print("\r" + " " * 35 + "\r", end="", flush=True)

    num_params = count_parameters(model)
    param_info = f"{num_params / 1e6:.1f}M ({format_number(num_params)})"

    hw_info = "CPU"
    if device.type == 'cuda':
        hw_info = f"GPU ({torch.cuda.get_device_name()})"

    print("=" * 70)
    print(r"""           ______                         ___    ____
          /_  __/__  _________ _____     /   |  /  _/
           / / / _ \/ ___/ __ `/_  /    / /| |  / /  
          / / /  __/ /  / /_/ / / /_   / ___ |_/ /   
         /_/  \___/_/   \__,_/ /___/  /_/  |_/___/   """)
    print("-" * 70)
    print(c("           ✦ TERAZ AI — Chatbot Indonesia ✦", Colors.BOLD))
    print("=" * 70)
    print(f"  Parameter  : {param_info}")
    print(f"  Dataset    : 3 Juta+ Baris")
    print(f"  Ditraining : 14 Jam menggunakan NVIDIA RTX 3050 6GB")
    print(f"  Hardware   : {hw_info}")
    print(f"  Repository : https://github.com/bimantaraz/teraz-ai-bhs-indonesia")
    print(f"  Status     : Siap digunakan")
    print(f"  Bantuan    : /help, /save, /temp, /reset, /exit")
    print("=" * 70)
    print()

    history_turns: List[List[int]] = []
    conversation_log: List[Tuple[str, str]] = []

    user_label = c("[Anda]     : ", Colors.BOLD_CYAN)
    ai_label = c("[Teraz AI] : ", Colors.BOLD_GREEN)

    while True:
        try:
            user_text = input(user_label).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{ai_label}Sampai jumpa!\n")
            break

        if not user_text:
            continue

        cmd = user_text.lower().strip()

        if cmd in ("/exit", "/quit"):
            print(f"\n{ai_label}Sampai jumpa!\n")
            break

        if cmd == "/help":
            print(c("\n  Daftar Perintah:", Colors.BOLD))
            print("    /help          - Menampilkan bantuan ini")
            print("    /save          - Menyimpan riwayat percakapan sesi ini ke file teks")
            print("    /temp <nilai>  - Mengubah temperature (contoh: /temp 0.8)")
            print("    /preset <nama> - Mengubah preset sampling (balanced / creative / precise)")
            print("    /reset         - Menghapus histori percakapan sesi ini")
            print("    /exit          - Keluar dari sesi chat\n")
            continue

        if cmd == "/save":
            if not conversation_log:
                print(c("  [Belum ada percakapan untuk disimpan]\n", Colors.YELLOW))
                continue
            fname = f"chat_teraz_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(f"=== Riwayat Chat Teraz AI ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")
                    for u, a in conversation_log:
                        f.write(f"[Anda]     : {u}\n")
                        f.write(f"[Teraz AI] : {a}\n\n")
                print(c(f"  [Obrolan berhasil disimpan ke '{fname}']\n", Colors.GREEN))
            except Exception as e:
                print(c(f"  [Gagal menyimpan: {e}]\n", Colors.YELLOW))
            continue

        if cmd.startswith("/temp"):
            parts = user_text.split()
            if len(parts) > 1:
                try:
                    val = float(parts[1])
                    temperature = max(0.01, min(2.0, val))
                    print(c(f"  [Temperature diubah menjadi {temperature:.2f}]\n", Colors.GREEN))
                except ValueError:
                    print(c("  [Format salah. Contoh penggunaan: /temp 0.8]\n", Colors.YELLOW))
            else:
                print(c(f"  [Temperature saat ini: {temperature:.2f}]\n", Colors.GREEN))
            continue

        if cmd.startswith("/preset"):
            parts = user_text.split()
            if len(parts) > 1 and parts[1].lower() in PRESETS:
                pname = parts[1].lower()
                p = PRESETS[pname]
                temperature = p["temperature"]
                top_k = p["top_k"]
                top_p = p["top_p"]
                repetition_penalty = p["repetition_penalty"]
                print(c(f"  [Preset aktif: '{pname}' ({p['desc']})]\n"
                        f"  (temp={temperature}, top_k={top_k}, top_p={top_p}, rep_pen={repetition_penalty})\n", Colors.GREEN))
            else:
                print(c("  [Pilihan preset: balanced, creative, precise. Contoh: /preset creative]\n", Colors.YELLOW))
            continue

        if cmd == "/reset":
            history_turns = []
            print(c("  [Histori percakapan direset]\n", Colors.YELLOW))
            continue

        turn_ids = [USER_ID] + tokenizer.encode(user_text).ids + [ASST_ID]

        context_ids = build_context(
            [] if args.no_history else history_turns,
            turn_ids, max_seq, args.max_new_tokens,
        )
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

        print(ai_label, end="", flush=True)

        try:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            gen_start = time.time()

            response_ids, reason = generate(
                model, tokenizer, input_ids, max_seq, device, amp_dtype,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stream=True,
            )

            if device.type == 'cuda':
                torch.cuda.synchronize()
            gen_elapsed = time.time() - gen_start
        except KeyboardInterrupt:
            print(c("\n  [Pesan dihentikan]\n", Colors.YELLOW))
            continue

        print("\n")

        if not response_ids:
            print(c("  (respons kosong — coba naikkan temperature)\n", Colors.YELLOW))
        else:
            n_tokens = len(response_ids)
            tps = n_tokens / max(gen_elapsed, 1e-9)
            note = ""
            if reason == "length":
                note = " | batas max_tokens"
            elif reason == "context":
                note = " | konteks penuh"
            stat_line = f"             ({n_tokens} token | {gen_elapsed:.1f}s | {tps:.1f} tok/s{note})"
            print(c(stat_line, Colors.GRAY) + "\n")

            full_resp = tokenizer.decode(response_ids)
            conversation_log.append((user_text, full_resp))

        if not args.no_history:
            history_turns.append(turn_ids + response_ids + [EOS_ID])

if __name__ == "__main__":
    main()
