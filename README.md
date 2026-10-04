# Teraz AI — Chatbot Berbahasa Indonesia

Teraz AI adalah model bahasa kausal (*Causal Language Model*) berbasis arsitektur Transformer Decoder-Only (~37.8 Juta parameter) yang dilatih dari nol (*from scratch*) menggunakan dataset 3jt+ baris percakapan Bahasa Indonesia.

Repositori ini berfokus pada **inferensi interaktif (chat)** yang sepenuhnya mandiri (*standalone*) dan ringan.

---

## Spesifikasi Model

| Parameter | Nilai | Keterangan |
| :--- | :--- | :--- |
| **Arsitektur** | Decoder-Only Transformer | Pre-LayerNorm, Scaled Dot-Product Attention |
| **Total Parameter** | ~37,770,240 (37.8M) | Weight-tied embedding & LM head |
| **Dimensi Model (`d_model`)** | 512 | Dimensi representasi tersembunyi |
| **Jumlah Layer (`n_layers`)** | 8 | Blok Transformer |
| **Attention Heads (`n_heads`)** | 8 | 64 dimensi per head |
| **Panjang Konteks Maksimal (`max_seq`)** | 512 token | Context window |
| **Ukuran Kosakata (`vocab_size`)** | 24,000 | Byte-Level BPE (aman dari token `<unk>`) |
| **Special Tokens** | `<pad>`, `<eos>`, `<unk>`, `<|user|>`, `<|assistant|>` | Penanda struktur percakapan |

---

## Struktur Proyek

Struktur proyek dibuat datar (*flat*) agar sederhana dan langsung siap dijalankan:

```text
teraz-ai-bhs-indonesia/
├── chat.py                 # Skrip chat interaktif mandiri (Inference)
├── tokenizer.json          # Byte-Level BPE tokenizer (~24k vocab)
├── model_config.json       # Konfigurasi arsitektur Transformer
├── preprocess_meta.json    # Metadata token dan dataset
├── requirements.txt        # Dependensi Python
├── .gitignore              # Mengabaikan file bobot besar (>100MB) & cache
├── README.md               # Dokumentasi proyek
└── Teraz-AI.pt             # Checkpoint bobot model (~453 MB) [Unduh via Releases/HuggingFace]
```

---

## Instalasi & Persiapan

### 1. Klon Repositori & Masuk ke Direktori
```bash
git clone https://github.com/bimantaraz/teraz-ai-bhs-indonesia.git
cd teraz-ai-bhs-indonesia
```

### 2. Buat Lingkungan Virtual & Instal Dependensi
```bash
# Buat virtual environment (disarankan)
python -m venv .venv

# Aktivasi di Windows:
.venv\Scripts\activate
# Atau di Linux/macOS:
source .venv/bin/activate

# Instal dependensi
pip install -r requirements.txt
```

> **Catatan GPU**: Jika perangkat Anda memiliki GPU NVIDIA, pastikan PyTorch dengan CUDA terpasang agar akselerasi inferensi berjalan maksimal.

---

## Contoh Tampilan & Interaksi Chat

Tampilan terminal saat [chat.py](chat.py) dijalankan:

```text
======================================================================
           ______                         ___    ____
          /_  __/__  _________ _____     /   |  /  _/
           / / / _ \/ ___/ __ `/_  /    / /| |  / /  
          / / /  __/ /  / /_/ / / /_   / ___ |_/ /   
         /_/  \___/_/   \__,_/ /___/  /_/  |_/___/   
----------------------------------------------------------------------
           ✦ TERAZ AI — Chatbot Indonesia ✦
======================================================================
  Parameter  : 37.8M (37.770.240)
  Dataset    : 3 Juta+ Baris
  Ditraining : 14 Jam Menggunakan GPU NVIDIA RTX 3050 6GB
  Hardware   : GPU (NVIDIA GeForce RTX 5090)
  Repository : https://github.com/bimantaraz/teraz-ai-bhs-indonesia
  Status     : Siap digunakan
  Bantuan    : /help, /save, /temp, /reset, /exit
======================================================================

[Anda]     : Halo! Siapa kamu dan apa kemampuanmu?
[Teraz AI] : Halo! Saya Teraz AI, asisten virtual buatan anak bangsa berbahasa Indonesia. Saya dapat membantu menjawab pertanyaan, berdiskusi, serta membuat teks kreatif. Ada yang bisa saya bantu hari ini?
             (41 token | 1.1s | 38.2 tok/s)
```

---

## Menjalankan Chat (`chat.py`)

Skrip `chat.py` dapat dijalankan secara langsung. Skrip secara otomatis mendeteksi file konfigurasi `model_config.json`, tokenizer `tokenizer.json`, dan bobot checkpoint `Teraz-AI.pt` di direktori yang sama.

### Penggunaan Dasar:
```bash
python chat.py
```

### Preset Sampling Cepat:
```bash
# Preset balanced (default) - Keseimbangan kreativitas dan akurasi
python chat.py --preset balanced

# Preset creative - Lebih bervariasi dan imajinatif (cerita, ide, puisi)
python chat.py --preset creative

# Preset precise - Lebih fokus, konsisten, dan faktual (fakta, pengetahuan)
python chat.py --preset precise
```

### Opsi & Parameter Tambahan:
```bash
# Mengatur nilai sampling secara manual (override preset)
python chat.py --temperature 0.7 --top-k 50 --top-p 0.9

# Mengatur penalti pengulangan kata (repetition penalty)
python chat.py --repetition-penalty 1.15

# Mengatur panjang token jawaban maksimum
python chat.py --max-new-tokens 256

# Mode tanpa histori (tiap prompt mandiri)
python chat.py --no-history
```

### Perintah dalam Sesi Chat:
- `/help` : Menampilkan daftar bantuan perintah.
- `/save` : Menyimpan riwayat percakapan sesi ini ke file teks (`chat_teraz_*.txt`).
- `/temp <nilai>` : Mengubah temperature sampling secara langsung (contoh: `/temp 0.8`).
- `/preset <nama>` : Mengubah preset sampling on-the-fly (`balanced`, `creative`, `precise`).
- `/reset` : Menghapus histori konteks percakapan sesi ini.
- `/exit` atau `/quit` : Keluar dari program (atau tekan `Ctrl + C`).

---

## Panduan Upload ke GitHub

GitHub memiliki batasan ukuran file maksimal **100 MB** per file. File bobot `Teraz-AI.pt` berukuran **~453 MB**, sehingga **tidak boleh di-push langsung** ke commit reguler Git. File ini sudah otomatis diabaikan oleh `.gitignore`.

### 1. Push Kode ke Repository GitHub:
```bash
git add .
git commit -m "feat: standalone chat, tokenizer, and model configs"
git branch -M main
git remote add origin https://github.com/bimantaraz/teraz-ai-bhs-indonesia.git
git push -u origin main
```

### 2. Menyediakan File Bobot Model (`Teraz-AI.pt`):

Pilih salah satu metode berikut untuk membagikan file model kepada pengguna:

- **Opsi A — GitHub Releases (Sangat Direkomendasikan)**:
  1. Buka halaman repo GitHub Anda, klik menu **Releases** di sisi kanan > **Draft a new release**.
  2. Masukkan tag versi (misal: `v1.0.0`).
  3. Drag & drop file `Teraz-AI.pt` ke area **Attach binaries by dropping them here** (GitHub Releases mendukung file hingga 2 GB per file secara gratis).
  4. Pengguna cukup mengunduh file `Teraz-AI.pt` tersebut dan meletakkannya di folder utama proyek (sejajar dengan `chat.py`).

- **Opsi B — Hugging Face Model Hub**:
  1. Buat repository model baru di [Hugging Face](https://huggingface.co/new).
  2. Unggah file `Teraz-AI.pt`, `tokenizer.json`, dan `model_config.json`.
  3. Berikan tautan model Hugging Face Anda pada README ini agar pengguna dapat mengunduhnya dengan mudah.

---

## Lisensi
Proyek ini memiliki lisensi [MIT License](LICENSE).
