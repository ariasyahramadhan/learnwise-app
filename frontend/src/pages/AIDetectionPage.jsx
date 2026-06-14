import { useState, useRef } from "react";
import { Cpu, CloudUpload, Trash, ArrowLeft, Edit, Clipboard } from "../components/Icons";
import "./AIDetectionPage.css";

export default function AIDetectionPage() {
  const [activeMode, setActiveMode] = useState("text"); // "text" or "file"
  const [inputText, setInputText] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const API_URL = import.meta.env.VITE_AI_DETECTION_API_URL 
             || import.meta.env.VITE_API_URL 
             || "http://localhost:8001";

  const fileInputRef = useRef(null);

  const handleReset = () => {
    setInputText("");
    setSelectedFile(null);
    setResults(null);
    setError(null);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Validate file type
    const allowed = [".pdf", ".docx", ".txt"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext)) {
      setError("Format file tidak didukung. Hanya PDF, DOCX, dan TXT.");
      setSelectedFile(null);
      return;
    }

    setError(null);
    setSelectedFile(file);
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;

    const allowed = [".pdf", ".docx", ".txt"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext)) {
      setError("Format file tidak didukung. Hanya PDF, DOCX, dan TXT.");
      setSelectedFile(null);
      return;
    }

    setError(null);
    setSelectedFile(file);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleDetectAI = async (e) => {
    e.preventDefault();
    
    if (activeMode === "text" && !inputText.trim()) {
      setError("Masukkan teks terlebih dahulu.");
      return;
    }
    if (activeMode === "file" && !selectedFile) {
      setError("Pilih atau seret dokumen terlebih dahulu.");
      return;
    }

    setIsDetecting(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    if (activeMode === "text") {
      formData.append("text", inputText);
    } else {
      formData.append("file", selectedFile);
    }

    try {
        const response = await fetch(`${API_URL}/api/detect-ai`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Gagal melakukan deteksi.");
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || "Gagal menghubungkan ke server.");
    } finally {
      setIsDetecting(false);
    }
  };

  // Determine gauge color and descriptor based on AI probability
  const getAIProfile = (prob) => {
    if (prob >= 70) {
      return {
        color: "var(--danger)",
        bg: "rgba(239, 68, 68, 0.1)",
        label: "Sangat Mungkin Buatan AI",
        desc: "Teks ini menunjukkan pola tulisan yang sangat khas buatan model AI / LLM (seperti ChatGPT). Gaya kalimat sangat terstruktur, kosakata seragam, dan banyak menggunakan kata penghubung transisi AI."
      };
    } else if (prob >= 35) {
      return {
        color: "var(--warning)",
        bg: "rgba(245, 158, 11, 0.1)",
        label: "Kemungkinan Campuran / Parafrasa AI",
        desc: "Teks ini menunjukkan pola campuran. Terdapat beberapa kalimat yang kemungkinan ditulis oleh manusia, namun didekatkan dengan parafrasa atau bantuan alat AI generator."
      };
    } else {
      return {
        color: "var(--success)",
        bg: "rgba(16, 185, 129, 0.1)",
        label: "Ditulis oleh Manusia",
        desc: "Teks ini diidentifikasi ditulis secara manual oleh manusia. Pola variasi panjang kalimat, kekayaan kosa kata, dan penggunaan tanda baca menunjukkan karakteristik organik tulisan manusia."
      };
    }
  };

  const formatByteSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="ai-detection-page">
      {/* Header */}
      <div className="ai-header">
        <h2 className="ai-title">
          <Cpu size={24} style={{ color: "var(--primary)" }} />
          <span>Detektor Teks AI (AI Text Detector)</span>
        </h2>
        <p className="ai-subtitle">
          Analisis dokumen atau teks Bahasa Indonesia Anda untuk mengetahui probabilitas apakah konten tersebut buatan kecerdasan buatan (AI) atau tulisan manusia asli.
        </p>
      </div>

      {error && (
        <div className="error-box">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {!results ? (
        <form className="ai-form" onSubmit={handleDetectAI}>
          {/* Mode Switcher Tabs */}
          <div className="mode-tabs">
            <button
              type="button"
              className={`mode-tab ${activeMode === "text" ? "active" : ""}`}
              onClick={() => {
                setActiveMode("text");
                setError(null);
              }}
              disabled={isDetecting}
            >
              <Edit size={16} />
              <span>Input Teks Langsung</span>
            </button>
            <button
              type="button"
              className={`mode-tab ${activeMode === "file" ? "active" : ""}`}
              onClick={() => {
                setActiveMode("file");
                setError(null);
              }}
              disabled={isDetecting}
            >
              <CloudUpload size={16} />
              <span>Upload Dokumen File</span>
            </button>
          </div>

          {/* Form Inputs */}
          {activeMode === "text" ? (
            <div className="input-card text-input-card">
              <label className="input-label">
                <Clipboard size={16} style={{ color: "var(--primary)" }} />
                <span>Salin & Tempel Teks Anda</span>
              </label>
              <textarea
                className="ai-textarea"
                placeholder="Masukkan atau tempel teks Bahasa Indonesia yang ingin Anda analisis di sini (minimal 10 karakter)..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                maxLength={10000}
                disabled={isDetecting}
              />
              <div className="textarea-footer">
                <span className="char-counter">{inputText.length} / 10.000 karakter</span>
                {inputText && (
                  <button
                    type="button"
                    className="btn-clear-text"
                    onClick={() => setInputText("")}
                    disabled={isDetecting}
                  >
                    Kosongkan Teks
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="file-upload-card">
              <div
                className={`dropzone ${isDragging ? "dragging" : ""} ${selectedFile ? "has-file" : ""}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !selectedFile && fileInputRef.current.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  accept=".pdf,.docx,.txt"
                  onChange={handleFileChange}
                  style={{ display: "none" }}
                  disabled={isDetecting}
                />

                {!selectedFile ? (
                  <div className="dropzone-prompt">
                    <CloudUpload size={48} className="upload-icon" />
                    <h4>Seret dan lepaskan file dokumen Anda di sini</h4>
                    <p>atau klik untuk memilih file dari komputer</p>
                    <span className="file-formats">Mendukung format: .pdf, .docx, .txt (Maks 20MB)</span>
                  </div>
                ) : (
                  <div className="selected-file-display" onClick={(e) => e.stopPropagation()}>
                    <div className="file-info-icon">📄</div>
                    <div className="file-details">
                      <span className="file-name">{selectedFile.name}</span>
                      <span className="file-size">{formatByteSize(selectedFile.size)}</span>
                    </div>
                    <button
                      type="button"
                      className="btn-remove-selected-file"
                      onClick={handleRemoveFile}
                      disabled={isDetecting}
                      title="Hapus file"
                    >
                      <Trash size={16} />
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          <button
            type="submit"
            className={`btn-detect-action ${isDetecting ? "loading" : ""}`}
            disabled={isDetecting || (activeMode === "text" && !inputText.trim()) || (activeMode === "file" && !selectedFile)}
          >
            {isDetecting ? (
              <>
                <div className="spinner" />
                <span>Menganalisis Teks (Memindai Pola AI)...</span>
              </>
            ) : (
              <>
                <Cpu size={20} />
                <span>Mulai Deteksi Teks AI</span>
              </>
            )}
          </button>
        </form>
      ) : (
        <div className="ai-results-panel">
          <div className="results-panel-header">
            <div>
              <h3 className="results-panel-title">Hasil Deteksi Teks AI</h3>
              <p className="results-panel-desc">
                Analisis selesai berdasarkan {results.word_count} kata ({results.char_count} karakter).
              </p>
            </div>
            <button className="btn-reset-ai" onClick={handleReset}>
              <ArrowLeft size={14} />
              <span>Deteksi Teks Baru</span>
            </button>
          </div>

          <div className="results-grid">
            {/* Left Column: Gauge and Classification Result */}
            <div className="result-main-card">
              {/* Circular Gauge */}
              <div className="gauge-section">
                <div className="gauge-outer-large">
                  <svg className="gauge-svg-large" viewBox="0 0 120 120">
                    <circle
                      className="gauge-bg-circle-large"
                      cx="60"
                      cy="60"
                      r="52"
                    />
                    <circle
                      className="gauge-fill-circle-large"
                      cx="60"
                      cy="60"
                      r="52"
                      strokeDasharray={326.7}
                      strokeDashoffset={326.7 - (326.7 * results.ai_probability) / 100}
                      style={{ stroke: getAIProfile(results.ai_probability).color }}
                    />
                  </svg>
                  <div className="gauge-text-wrap-large">
                    <span className="gauge-score-val-large">{results.ai_probability}%</span>
                    <span className="gauge-score-label-large">KEMUNGKINAN AI</span>
                  </div>
                </div>
              </div>

              {/* Classification Tag & Explanation */}
              <div className="classification-details">
                <div
                  className="classification-badge"
                  style={{
                    backgroundColor: getAIProfile(results.ai_probability).bg,
                    color: getAIProfile(results.ai_probability).color
                  }}
                >
                  {getAIProfile(results.ai_probability).label}
                </div>
                <p className="classification-desc-text">
                  {getAIProfile(results.ai_probability).desc}
                </p>
              </div>
            </div>

            {/* Right Column: Stylometric Metrics Analysis */}
            <div className="result-metrics-card">
              <h4 className="metrics-card-title">🔍 Hasil Analisis Stylometry Teks</h4>
              <p className="metrics-card-desc">
                Berikut adalah karakteristik struktural teks Anda dibandingkan dengan rata-rata kecenderungan penulisan kecerdasan buatan (AI):
              </p>

              <div className="ai-metrics-list">
                
                {/* 1. Lexical Diversity (TTR) */}
                <div className="ai-metric-item">
                  <div className="metric-meta">
                    <span className="metric-label">Kepadatan Kosakata (TTR)</span>
                    <span className="metric-value">{results.metrics.lexical_diversity_ttr}</span>
                  </div>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill"
                      style={{
                        width: `${Math.min(results.metrics.lexical_diversity_ttr * 100, 100)}%`,
                        backgroundColor: results.metrics.lexical_diversity_ttr < 0.45 ? "var(--danger)" : "var(--success)"
                      }}
                    />
                  </div>
                  <p className="metric-help">
                    * Type-Token Ratio. AI cenderung menggunakan kosakata yang lebih seragam dan berulang (TTR rendah, di bawah 0.50).
                  </p>
                </div>

                {/* 2. Transition Word Rate */}
                <div className="ai-metric-item">
                  <div className="metric-meta">
                    <span className="metric-label">Kepadatan Kata Hubung Transisi AI</span>
                    <span className="metric-value">{results.metrics.transition_word_rate}%</span>
                  </div>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill"
                      style={{
                        width: `${Math.min(results.metrics.transition_word_rate * 20, 100)}%`,
                        backgroundColor: results.metrics.transition_word_rate > 2.5 ? "var(--danger)" : "var(--success)"
                      }}
                    />
                  </div>
                  <p className="metric-help">
                    * Persentase penggunaan kata hubung transisi khas LLM (seperti: *selain itu, oleh karena itu, namun, secara keseluruhan*).
                  </p>
                </div>

                {/* 3. Word and Sentence Lengths */}
                <div className="metrics-stats-grid">
                  <div className="stat-box">
                    <span className="stat-label">Panjang Kata (Rerata)</span>
                    <strong className="stat-val">{results.metrics.avg_word_length}</strong>
                    <span className="stat-sub">karakter per kata</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Panjang Kalimat (Rerata)</span>
                    <strong className="stat-val">{results.metrics.avg_sentence_length}</strong>
                    <span className="stat-sub">kata per kalimat</span>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Kepadatan Tanda Baca</span>
                    <strong className="stat-val">{results.metrics.punctuation_rate}</strong>
                    <span className="stat-sub">per 1000 karakter</span>
                  </div>
                </div>

              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
