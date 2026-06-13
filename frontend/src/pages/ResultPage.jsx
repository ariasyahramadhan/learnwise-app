import { useState } from "react";
import SimilarityTable from "../components/SimilarityTable";
import DocumentPlot from "../components/DocumentPlot";
import SimilarityMatrix from "../components/SimilarityMatrix";
import CompareModal from "../components/CompareModal";
import { Flask, Feather, Sliders, ChartBar, Map, Thermometer, Folder, Clipboard } from "../components/Icons";
import "./ResultPage.css";

const MODEL_LABELS = {
  model_a: { icon: Flask, name: "Surface Similarity (Model A · XGBoost)" },
  model_b: { icon: Feather, name: "Semantic Similarity (Model B · SBERT + XGBoost)" },
  model_c: { icon: Sliders, name: "Custom Hybrid (Model C · Kustom)" },
};

export default function ResultPage({ result, onReset }) {
  const [activeComparePair, setActiveComparePair] = useState(null);
  const { summary, documents, pairs, model } = result;
  const modelInfo = MODEL_LABELS[model] || MODEL_LABELS.model_a;
  const ModelIcon = modelInfo.icon;
  
  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
  
  const riskLevel = (val) => {
    if (val >= 80) return "high";
    if (val >= 50) return "medium";
    return "low";
  };

  return (
    <div className="result-page">
      {/* Header */}
      <div className="result-header">
        <h2 className="result-title" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <ChartBar size={24} style={{ color: "var(--primary)" }} />
          <span>Hasil Analisis Plagiarisme</span>
        </h2>
        <div className="model-used-tag" style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}>
          <ModelIcon size={16} />
          <span>Model: {modelInfo.name}</span>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="summary-grid">
        <div className="summary-card">
          <div className="summary-label">Total Dokumen</div>
          <div className="summary-value">{summary.total_documents}</div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Total Pasangan</div>
          <div className="summary-value">{summary.total_pairs}</div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Kemiripan Tertinggi</div>
          <div className={`summary-value ${riskLevel(summary.highest_similarity)}`}>
            {summary.highest_similarity}%
          </div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Rata-rata Kemiripan</div>
          <div className={`summary-value ${riskLevel(summary.average_similarity)}`}>
            {summary.average_similarity}%
          </div>
        </div>
      </div>

      {/* Document Plot */}
      <div className="section-card">
        <div className="section-header">
          <p className="section-title" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span className="section-title-icon" style={{ display: "flex", alignItems: "center" }}>
              <Map size={18} />
            </span>
            <span>Visualisasi Posisi Dokumen (2D)</span>
          </p>
        </div>
        <p className="section-desc">
          Dokumen yang <strong>berdekatan</strong> = lebih mirip. Posisi dihasilkan dari SBERT embeddings
          yang direduksi dengan {documents.length >= 5 ? "UMAP" : "PCA"}.
        </p>
        <DocumentPlot
          documents={documents}
          coordinates={result.coordinates_2d}
          pairs={pairs}
        />
      </div>

      {/* Pairwise Table */}
      <div className="section-card">
        <div className="section-header">
          <p className="section-title" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span className="section-title-icon" style={{ display: "flex", alignItems: "center" }}>
              <Clipboard size={18} />
            </span>
            <span>Perbandingan Antar Pasangan</span>
          </p>
        </div>
        <SimilarityTable pairs={pairs} onCompare={setActiveComparePair} />
      </div>

      {/* Heatmap */}
      <div className="section-card">
        <div className="section-header">
          <p className="section-title" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span className="section-title-icon" style={{ display: "flex", alignItems: "center" }}>
              <Thermometer size={18} />
            </span>
            <span>Matriks Kemiripan</span>
          </p>
        </div>
        <SimilarityMatrix matrix={result.similarity_matrix} documents={documents} onCompare={setActiveComparePair} />
      </div>

      {/* Documents */}
      <div className="section-card">
        <div className="section-header">
          <p className="section-title" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span className="section-title-icon" style={{ display: "flex", alignItems: "center" }}>
              <Folder size={18} />
            </span>
            <span>Detail Dokumen</span>
          </p>
        </div>
        <div className="doc-grid">
          {documents.map((doc, i) => (
            <div key={i} className="doc-card">
              <div className="doc-index">#{i + 1}</div>
              <div className="doc-name">{doc.name}</div>
              <div className="doc-meta">{doc.word_count?.toLocaleString()} kata</div>
              <p className="doc-preview">{doc.preview}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Comparison Modal Overlay */}
      {activeComparePair && (
        <CompareModal
          pair={activeComparePair}
          onClose={() => setActiveComparePair(null)}
          documents={documents}
        />
      )}
    </div>
  );
}
