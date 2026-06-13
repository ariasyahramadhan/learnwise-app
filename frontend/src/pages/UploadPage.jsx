import { useState, useRef, useCallback } from "react";
import { Flask, Feather, Sliders, CloudUpload, Search, Trash, Close, FilePdf, FileDoc, File, Folder } from "../components/Icons";
import "./UploadPage.css";

const MODELS = [
  {
    id: "model_a",
    icon: Flask,
    name: "Surface Similarity",
    desc: "Deteksi berbasis 26 fitur leksikal (TF-IDF Word + Char, N-Gram, Word Order, LCS, Edit Distance, dll.). Dilatih pada PAWS-Indonesia dengan XGBoost.",
    badge: "Model A · XGBoost",
  },
  {
    id: "model_b",
    icon: Feather,
    name: "Semantic Similarity",
    desc: "Deteksi kesamaan makna/arti kalimat secara kontekstual menggunakan embedding SBERT (Sentence-BERT) dan pengklasifikasi XGBoost.",
    badge: "Model B · SBERT + XGBoost",
  },
  {
    id: "model_c",
    icon: Sliders,
    name: "Custom Hybrid (Kombinasi)",
    desc: "Kombinasi kustom antara Surface Similarity (Model A) dan Stylometric Similarity (Model B) dengan bobot kustom.",
    badge: "Model C · Kustom",
  },
];

export default function UploadPage({ onResult, isAnalyzing, setIsAnalyzing }) {
  const [files, setFiles] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");
  const [selectedModel, setSelectedModel] = useState("model_a");
  const [weightA, setWeightA] = useState(50);
  const fileInputRef = useRef();

  const MAX_FILES = 50;

  const addFiles = useCallback((newFiles) => {
    const allowed = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/plain",
    ];
    const filtered = Array.from(newFiles).filter((f) => allowed.includes(f.type));
    if (filtered.length !== Array.from(newFiles).length) {
      setError("Beberapa file diabaikan. Hanya PDF, DOCX, dan TXT yang didukung.");
    }
    setFiles((prev) => {
      const combined = [...prev, ...filtered];
      if (combined.length > MAX_FILES) {
        setError(`Maksimal ${MAX_FILES} dokumen.`);
        return combined.slice(0, MAX_FILES);
      }
      return combined;
    });
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      setError("");
      addFiles(e.dataTransfer.files);
    },
    [addFiles]
  );

  const handleFileInput = (e) => {
    setError("");
    addFiles(e.target.files);
    e.target.value = "";
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setError("");
  };

  const handleAnalyze = async () => {
    if (files.length < 2) {
      setError("Upload minimal 2 dokumen untuk dibandingkan.");
      return;
    }
    setError("");
    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));
      formData.append("model", selectedModel);
      if (selectedModel === "model_c") {
        formData.append("weight_a", (weightA / 100).toString());
        formData.append("weight_b", ((100 - weightA) / 100).toString());
      }
      
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      
      const res = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(120000),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Terjadi kesalahan pada server");
      }

      const data = await res.json();
      onResult({ ...data, model: selectedModel });
    } catch (err) {
      if (err.name === "TimeoutError") {
        setError("Waktu habis. Coba lagi dengan lebih sedikit dokumen.");
      } else {
        setError(err.message);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
  };

  const getFileIcon = (type) => {
    if (type === "application/pdf") return <FilePdf size={24} style={{ color: "var(--danger)", marginRight: "8px" }} />;
    if (type.includes("wordprocessingml")) return <FileDoc size={24} style={{ color: "var(--info)", marginRight: "8px" }} />;
    return <File size={24} style={{ color: "var(--text-dim)", marginRight: "8px" }} />;
  };

  return (
    <div className="upload-page">
      {/* Hero */}
      <div className="upload-hero">
        <div className="hero-eyebrow" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.4rem" }}>
          <Search size={14} />
          <span>Deteksi Plagiarisme</span>
        </div>
        <h1 className="hero-title">
          Analisis Kemiripan <span className="accent">Dokumen</span>
        </h1>
        <p className="hero-desc">
          Upload 2–{MAX_FILES} dokumen (PDF, DOCX, TXT). LearnWise akan membandingkan kemiripan antar
          dokumen menggunakan model bahasa <em style={{ fontStyle: "normal", color: "var(--primary)" }}>multilingual</em>.
        </p>
      </div>

      {/* Model Selector */}
      <div className="model-selector-section">
        <div className="model-selector-label" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <Sliders size={18} />
          <span>Pilih Model Deteksi</span>
        </div>
        <div className="model-cards">
          {MODELS.map((m) => {
            const ModelIcon = m.icon;
            return (
              <button
                key={m.id}
                className={`model-card ${selectedModel === m.id ? "selected" : ""}`}
                onClick={() => setSelectedModel(m.id)}
              >
                <span className="model-card-icon">
                  <ModelIcon size={24} />
                </span>
                <div className="model-card-name">{m.name}</div>
                <div className="model-card-desc">{m.desc}</div>
                <div className="model-badge">{m.badge}</div>
              </button>
            );
          })}
        </div>

        {selectedModel === "model_c" && (
          <div className="hybrid-weight-slider">
            <div className="slider-label" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Sliders size={18} />
              <span>Atur Kombinasi Bobot Model C</span>
            </div>
            <div className="slider-container">
              <div className="weight-display-left">
                Model A (Surface): <strong>{weightA}%</strong>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={weightA}
                onChange={(e) => setWeightA(parseInt(e.target.value))}
                className="weight-slider"
              />
              <div className="weight-display-right">
                Model B (Semantic): <strong>{100 - weightA}%</strong>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Dropzone */}
      <div
        className={`dropzone ${dragOver ? "drag-over" : ""}`}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => fileInputRef.current.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt"
          style={{ display: "none" }}
          onChange={handleFileInput}
        />
        <div className="drop-icon-wrap" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <CloudUpload size={48} className="drop-icon-svg" />
        </div>
        <p className="drop-text">
          {dragOver ? "Lepaskan file di sini" : "Seret & lepas dokumen ke sini"}
        </p>
        <p className="drop-sub">atau pilih file dari perangkat Anda</p>
        <div className="drop-btn" style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem" }}>
          <Folder size={16} />
          <span>Pilih Dokumen</span>
        </div>
        <p className="drop-hint">PDF · DOCX · TXT · Maks {MAX_FILES} file</p>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="file-list-section">
          <div className="file-list-header">
            <span className="file-list-title">
              Dokumen Terpilih
              <span className="file-count-badge">{files.length}/{MAX_FILES}</span>
            </span>
            <button className="btn-clear" onClick={() => setFiles([])} style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
              <Trash size={14} />
              <span>Hapus semua</span>
            </button>
          </div>
          <div className="file-items">
            {files.map((file, i) => (
              <div key={file.name + file.size + i} className="file-item">
                <span className="file-icon" style={{ display: "flex", alignItems: "center" }}>
                  {getFileIcon(file.type)}
                </span>
                <div className="file-info">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">{formatSize(file.size)}</span>
                </div>
                <button
                  className="btn-remove"
                  onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                  style={{ display: "flex", alignItems: "center", justifyContent: "center" }}
                >
                  <Close size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="error-box" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span style={{ fontSize: "1.1rem" }}>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      <button
        className={`btn-analyze ${isAnalyzing ? "loading" : ""}`}
        onClick={handleAnalyze}
        disabled={isAnalyzing || files.length < 2}
        style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem" }}
      >
        {isAnalyzing ? (
          <><span className="spinner" />Menganalisis dokumen...</>
        ) : (
          <>
            <Search size={18} />
            <span>Mulai Analisis {files.length > 0 ? `(${files.length} dokumen)` : ""}</span>
          </>
        )}
      </button>

      {isAnalyzing && (
        <div className="analyzing-steps">
          <div className="step"><span className="step-spinner" /> Mengekstrak teks dari dokumen...</div>
          <div className="step" style={{ animationDelay: "0.3s", opacity: 0 }}>
            <span className="step-spinner" /> Mengekstrak fitur leksikal & gaya penulisan...
          </div>
          <div className="step" style={{ animationDelay: "0.6s", opacity: 0 }}>
            <span className="step-spinner" /> Menghitung skor kemiripan & reduksi dimensi...
          </div>
        </div>
      )}
    </div>
  );
}
