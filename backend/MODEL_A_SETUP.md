# Setup Model A (XGBoost Pipeline)

Model A menggunakan pipeline yang dilatih pada notebook `Model_A_ver1.ipynb`.
Ikuti langkah berikut agar model bisa digunakan di backend.

## Langkah 1 — Jalankan Notebook

Buka dan jalankan `Model_A_ver1.ipynb` dari awal hingga **Cell 20 (Simpan Pipeline)**.

Setelah Cell 20 selesai, akan terbuat folder `model_artifacts/` berisi:
```
model_artifacts/
├── tfidf_vectorizer.pkl
├── xgboost_classifier.pkl
└── pipeline_config.pkl
```

## Langkah 2 — Salin ke Backend

Salin folder `model_artifacts/` ke dalam folder `backend/`:

```
plagiarism-detector/
└── backend/
    ├── model_artifacts/          ← letakkan di sini
    │   ├── tfidf_vectorizer.pkl
    │   ├── xgboost_classifier.pkl
    │   └── pipeline_config.pkl
    ├── main.py
    ├── services.py
    └── requirements.txt
```

## Langkah 3 — Install Dependencies Baru

```bash
pip install -r requirements.txt
```

Dependencies baru yang ditambahkan:
- `joblib` — untuk memuat file `.pkl`
- `xgboost` — classifier utama Model A
- `PySastrawi` — stemmer Bahasa Indonesia (opsional, tetap jalan tanpa ini)
- `scipy` — untuk Word Order Score (Kendall tau)

## Langkah 4 — Jalankan Backend

```bash
uvicorn main:app --reload
```

Cek status Model A:
```
GET http://localhost:8000/api/model-status
```

Jika berhasil:
```json
{ "model_a_available": true, "message": "Model A siap digunakan." }
```

## Catatan

- Jika `model_artifacts/` belum ada, server **tetap bisa berjalan**.
  Model "Surface Similarity (Model A)" akan fallback ke NGram.
- Hybrid mode secara otomatis menggunakan Model A jika tersedia,
  NGram jika belum.
