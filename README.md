# LearnWise PlagiScan — Sistem Deteksi Plagiarisme Lanjut

LearnWise PlagiScan adalah aplikasi deteksi plagiarisme dokumen berbasis web yang menggabungkan metode analisis leksikal (**Surface Similarity**), analisis gaya penulisan (**Stylometric Similarity**), dan kecocokan makna semantik (**SBERT**). Aplikasi ini dibangun menggunakan **FastAPI** di sisi backend dan **React + Vite** di sisi frontend.

---

## 🌟 Fitur Utama

1. **Surface Similarity (Model A - Klasifikasi XGBoost):**
   - Mendeteksi plagiarisme berbasis kemiripan permukaan dokumen menggunakan model klasifikasi XGBoost yang dilatih dengan **26 fitur leksikal**.
   - Fitur mencakup: TF-IDF (word & char n-gram) Cosine Similarity, Jaccard Similarity, Longest Common Subsequence (LCS), Levenshtein Edit Distance, token sort, word order score, shared unique word ratio, prefix matches, dan statistik panjang teks.

2. **Stylometric Similarity (Model B - Analisis Gaya Penulisan):**
   - Mendeteksi plagiarisme berbasis gaya penulisan menggunakan model klasifikasi XGBoost dengan **16 fitur stilometri**.
   - Berguna untuk mendeteksi apakah suatu dokumen ditulis oleh orang yang sama atau ditulis ulang dengan gaya bahasa yang mirip.
   - Fitur mencakup: Type-Token Ratio (TTR/kepadatan kosakata), rata-rata panjang kata & kalimat, frekuensi tanda baca (koma, titik, titik dua, tanda tanya), distribusi stopword bahasa Indonesia (`yang`, `dan`, `di`, `ke`, `dari`, `untuk`), serta rasio huruf besar dan angka.

3. **Custom Hybrid (Model C - Kombinasi Kustom):**
   - Memungkinkan pengguna menggabungkan Model A (Surface) dan Model B (Stylometric) menggunakan **slider interaktif** di frontend untuk menentukan bobot persentase kontribusi masing-masing model (misalnya: 70% Model A dan 30% Model B).

4. **Cek Kemiripan Berdampingan (Side-by-Side Comparison):**
   - Fitur analisis kalimat terperinci yang menampilkan kedua dokumen secara berdampingan dalam panel independen.
   - Kalimat yang mirip terdeteksi secara instan di sisi klien menggunakan Jaccard similarity (threshold 0.35) dan disorot dengan warna kuning lembut.
   - **Hover Sinkron:** Mengarahkan kursor ke kalimat yang disorot pada dokumen kiri akan secara otomatis menyoroti kalimat pasangannya di dokumen kanan dengan warna emas/oranye gelap, memudahkan pelacakan kalimat plagiat/parafrasa.
   - **Kinerja Cepat:** Analisis kalimat berjalan langsung di browser klien dalam **1–5 ms** dengan memanfaatkan teks dokumen yang disertakan pada payload respon API awal, sehingga tidak membebani server dan tidak ada delay jaringan saat tombol ditekan.

5. **Matriks Kemiripan Interaktif (Similarity Heatmap):**
   - Visualisasi matriks korelasi kemiripan antar dokumen yang intuitif dengan pewarnaan gradien berbasis tingkat risiko plagiarisme (Merah: Tinggi, Kuning/Oranye: Sedang, Hijau: Rendah).
   - Layout kolom header mengalir secara horizontal dengan rapi tanpa masalah penumpukan vertikal.
   - Mengeklik sel matriks mana saja (selain sel diagonal) akan langsung memicu modal *Side-by-Side Comparison*.

6. **Visualisasi Posisi Dokumen 2D:**
   - Memetakan dokumen ke dalam grafik koordinat 2D menggunakan embedding **SBERT** (`paraphrase-multilingual-MiniLM-L12-v2`) yang direduksi dimensinya menggunakan algoritma **UMAP** atau **PCA** (untuk dokumen < 5). Dokumen yang berdekatan pada plot mengindikasikan kemiripan topik/makna.

---

## 🛠️ Stack Teknologi

| Layer | Teknologi |
|-------|-----------|
| **Backend** | FastAPI, Uvicorn, Python 3.10+ |
| **Machine Learning** | XGBoost Classifier, Scikit-learn, PySastrawi (opsional) |
| **Representasi Teks** | Sentence-Transformers (SBERT), TF-IDF (Word & Char) |
| **Reduksi Dimensi** | UMAP, PCA (sklearn) |
| **Parsing File** | pdfplumber (PDF), python-docx (DOCX) |
| **Frontend** | React, Vite, CSS Vanilla (Custom Glassmorphism & Micro-animations) |

---

## 📁 Struktur Proyek (Reorganisasi)

Struktur direktori telah dirapikan untuk memisahkan logika pelatihan model, file model biner, dan kode aplikasi:

```
plagiarism-detector/
├── backend/
│   ├── main.py                          # FastAPI Entrypoint & Routing API
│   ├── services.py                      # Logika utama ekstraksi teks & PlagiarismService
│   ├── requirements.txt                 # Dependensi Python backend
│   └── model_artifacts/                 # Penyimpanan file biner model hasil training
│       ├── surface_similarity/          # Artefak Model A
│       │   ├── surface_word_vectorizer.pkl
│       │   ├── surface_char_vectorizer.pkl
│       │   ├── surface_classifier.pkl
│       │   └── surface_config.pkl
│       └── stylometric_similarity/      # Artefak Model B
│           ├── stylometric_classifier.pkl
│           └── stylometric_config.pkl
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                      # Kontainer Utama & Navigasi
│       ├── App.css                      # Styling & Design Tokens Global
│       ├── main.jsx                     # Render React Root
│       ├── pages/
│       │   ├── UploadPage.jsx           # Dashboard unggah dokumen & set parameter model
│       │   ├── UploadPage.css
│       │   ├── ResultPage.jsx           # Dashboard visualisasi hasil deteksi
│       │   └── ResultPage.css
│       └── components/
│           ├── Icons.jsx                # Registri ikon SVG kustom (feather-like)
│           ├── CompareModal.jsx         # Modal analisis kemiripan side-by-side
│           ├── CompareModal.css
│           ├── SimilarityTable.jsx      # Tabel perbandingan berpasangan
│           ├── SimilarityTable.css
│           ├── SimilarityMatrix.jsx     # Heatmap matriks kemiripan
│           ├── SimilarityMatrix.css
│           ├── DocumentPlot.jsx         # Grafik sebaran 2D dokumen
│           └── DocumentPlot.css
├── notebooks/                           # Dokumentasi & Log Pelatihan Model
│   ├── surface_similarity_training.ipynb      # Notebook training Model A (Surface)
│   ├── stylometric_similarity_training.ipynb  # Notebook training Model B (Stylometric)
│   ├── archive/                         # Versi notebook Model A lama (ver1 - ver7)
│   ├── plots/                           # Grafik visualisasi hasil training (ROC, Confusion Matrix, dll.)
│   └── src/                             # Script pelatihan Python standalone
│       └── Model_A_ver8_optimized.py
├── .gitignore                           # Konfigurasi pengecualian Git
└── README.md                            # Dokumentasi sistem
```

---

## 🚀 Panduan Menjalankan Aplikasi

### 1. Prasyarat
Pastikan komputer Anda telah terinstal **Python 3.10+** dan **Node.js (LTS)**.

### 2. Menjalankan Backend (API FastAPI)
1. Masuk ke direktori backend:
   ```bash
   cd backend
   ```
2. Buat virtual environment & aktifkan:
   ```bash
   python -m venv venv
   # Di Windows (PowerShell/CMD):
   venv\Scripts\activate
   # Di Linux/Mac:
   source venv/bin/activate
   ```
3. Install semua dependensi backend:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan server backend:
   ```bash
   python main.py
   ```
   *Server backend akan berjalan secara default di `http://localhost:8000` dengan hot-reload aktif.*

> 💡 **Info:** Pada saat pertama kali dijalankan, backend akan mengunduh model SBERT `paraphrase-multilingual-MiniLM-L12-v2` (~120MB) dari HuggingFace secara otomatis.

### 3. Menjalankan Frontend (Vite + React)
1. Buka terminal baru dan masuk ke direktori frontend:
   ```bash
   cd frontend
   ```
2. Install dependensi frontend:
   ```bash
   npm install
   ```
3. Jalankan server development:
   ```bash
   npm run dev
   ```
4. Buka peramban (browser) Anda ke alamat **`http://localhost:5173`**.

---

## 📈 Parameter & Batasan Analisis

- **Format File yang Didukung:** PDF (`.pdf`), Word (`.docx`), dan Plain Text (`.txt`).
- **Batasan File:** Maksimal **50 dokumen** sekaligus dalam satu kali analisis, dengan ukuran file maksimal **20MB** per file.
- **Minimum input:** Minimal **2 dokumen** diperlukan untuk melakukan analisis perbandingan.
