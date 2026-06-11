# PlagiScan — Deteksi Plagiarisme Dokumen

Web app deteksi plagiarisme menggunakan **FastAPI** (backend) + **React/Vite** (frontend).

## Stack Teknologi

| Layer | Teknologi |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| ML Model | SBERT `paraphrase-multilingual-MiniLM-L12-v2` |
| Similarity | Cosine Similarity (sklearn) |
| Visualisasi 2D | UMAP / PCA |
| Frontend | React + Vite |
| File Parsing | pdfplumber, python-docx |

---

## Cara Menjalankan

### 1. Backend

```bash
cd backend

# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau: venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Jalankan server
python main.py
# Server berjalan di http://localhost:8000
```

> ⚠️ Pertama kali run, SBERT akan download model ~120MB otomatis.

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Jalankan dev server
npm run dev
# Buka http://localhost:5173
```

---

## Struktur Project

```
plagiarism-detector/
├── backend/
│   ├── main.py          # FastAPI app & endpoints
│   ├── services.py      # PlagiarismService (ML logic)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── pages/
    │   │   ├── UploadPage.jsx   # Halaman upload dokumen
    │   │   └── ResultPage.jsx   # Halaman hasil analisis
    │   └── components/
    │       ├── DocumentPlot.jsx   # Plot 2D posisi dokumen
    │       ├── SimilarityTable.jsx # Tabel perbandingan pasangan
    │       └── SimilarityMatrix.jsx # Heatmap matrix
    ├── package.json
    └── vite.config.js
```

---

## Alur ML

```
Upload Dokumen (PDF/DOCX/TXT)
        ↓
Ekstrak Teks (pdfplumber / python-docx)
        ↓
SBERT Embedding (768 dimensi, multilingual)
        ↓
    ┌───────────────────────┐
    │                       │
Cosine Similarity      UMAP / PCA
    │                       │
Tabel % Kemiripan    Plot 2D Posisi Dokumen
```

---

## Format File yang Didukung

- PDF (`.pdf`)
- Word Document (`.docx`)
- Plain Text (`.txt`)
- Maksimal 10 file sekaligus
