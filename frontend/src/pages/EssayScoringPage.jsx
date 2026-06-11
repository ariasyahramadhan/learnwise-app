import { useState } from "react";
import { Edit, Clipboard, Sliders, ArrowLeft, Folder } from "../components/Icons";
import "./EssayScoringPage.css";

export default function EssayScoringPage() {
  const [studentAnswer, setStudentAnswer] = useState("");
  const [referenceAnswer, setReferenceAnswer] = useState("");
  const [isScoring, setIsScoring] = useState(false);
  const [scoreResult, setScoreResult] = useState(null);
  const [error, setError] = useState(null);

  const handleScoreEssay = async (e) => {
    e.preventDefault();
    if (!referenceAnswer.strip ? !referenceAnswer.trim() : !referenceAnswer.trim()) {
      setError("Kunci jawaban tidak boleh kosong.");
      return;
    }
    if (!studentAnswer.strip ? !studentAnswer.trim() : !studentAnswer.trim()) {
      setError("Jawaban siswa tidak boleh kosong.");
      return;
    }

    setIsScoring(true);
    setError(null);
    setScoreResult(null);

    try {
      const response = await fetch("http://localhost:8000/api/score-essay", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          student_answer: studentAnswer,
          reference_answer: referenceAnswer,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Terjadi kesalahan pada server.");
      }

      const data = await response.json();
      setScoreResult(data);
    } catch (err) {
      setError(err.message || "Gagal menghubungkan ke server.");
    } finally {
      setIsScoring(false);
    }
  };

  const handleReset = () => {
    setStudentAnswer("");
    setReferenceAnswer("");
    setScoreResult(null);
    setError(null);
  };

  const getGradeInfo = (score) => {
    if (score >= 85) return { grade: "A", color: "var(--success)", text: "Sangat Baik! Jawaban sangat akurat dan mencakup semua materi utama." };
    if (score >= 70) return { grade: "B", color: "var(--info)", text: "Baik! Jawaban sudah cukup akurat dan terstruktur dengan benar." };
    if (score >= 55) return { grade: "C", color: "var(--warning)", text: "Cukup! Terdapat beberapa materi penting yang terlewatkan atau kurang lengkap." };
    if (score >= 40) return { grade: "D", color: "var(--primary-dark)", text: "Kurang! Jawaban kurang mendalam dan memiliki kecocokan yang rendah." };
    return { grade: "E", color: "var(--danger)", text: "Sangat Kurang! Jawaban tidak sesuai dengan kunci jawaban atau terlalu sedikit." };
  };

  const gradeInfo = scoreResult ? getGradeInfo(scoreResult.score) : null;

  return (
    <div className="essay-scoring-page">
      {/* Header */}
      <div className="scoring-header">
        <h2 className="scoring-title">
          <Edit size={24} style={{ color: "var(--primary)" }} />
          <span>Penilaian Esai Otomatis (Essay Scoring)</span>
        </h2>
        <p className="scoring-subtitle">
          Bandingkan jawaban siswa secara semantik dan leksikal dengan kunci jawaban untuk penilaian cepat.
        </p>
      </div>

      {error && (
        <div className="error-box">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {!scoreResult ? (
        <form className="scoring-form" onSubmit={handleScoreEssay}>
          <div className="input-columns">
            {/* Left Column: Reference Answer */}
            <div className="input-card">
              <label className="input-label">
                <Clipboard size={16} style={{ color: "var(--primary)" }} />
                <span>Kunci Jawaban / Referensi (Reference Answer)</span>
              </label>
              <textarea
                className="scoring-textarea"
                placeholder="Masukkan jawaban yang benar atau kunci jawaban referensi di sini..."
                value={referenceAnswer}
                onChange={(e) => setReferenceAnswer(e.target.value)}
                maxLength={5000}
                disabled={isScoring}
              />
              <div className="char-counter">
                {referenceAnswer.length} / 5000 karakter
              </div>
            </div>

            {/* Right Column: Student Answer */}
            <div className="input-card">
              <label className="input-label">
                <Edit size={16} style={{ color: "var(--primary)" }} />
                <span>Jawaban Siswa (Student's Response)</span>
              </label>
              <textarea
                className="scoring-textarea"
                placeholder="Masukkan jawaban esai yang ditulis siswa untuk dinilai di sini..."
                value={studentAnswer}
                onChange={(e) => setStudentAnswer(e.target.value)}
                maxLength={5000}
                disabled={isScoring}
              />
              <div className="char-counter">
                {studentAnswer.length} / 5000 karakter
              </div>
            </div>
          </div>

          <button
            type="submit"
            className={`btn-score ${isScoring ? "loading" : ""}`}
            disabled={isScoring || !referenceAnswer.trim() || !studentAnswer.trim()}
          >
            {isScoring ? (
              <>
                <div className="spinner" />
                <span>Sedang Menganalisis Jawaban...</span>
              </>
            ) : (
              <>
                <Sliders size={20} />
                <span>Mulai Penilaian Esai</span>
              </>
            )}
          </button>
        </form>
      ) : (
        <div className="scoring-result-container">
          <div className="result-main-card">
            {/* Left Result Side: Score Gauge */}
            <div className="score-gauge-side">
              <div className="gauge-outer">
                <svg className="gauge-svg" viewBox="0 0 100 100">
                  <circle
                    className="gauge-bg-circle"
                    cx="50"
                    cy="50"
                    r="40"
                  />
                  <circle
                    className="gauge-fill-circle"
                    cx="50"
                    cy="50"
                    r="40"
                    strokeDasharray={251.2}
                    strokeDashoffset={251.2 - (251.2 * scoreResult.score) / 100}
                    style={{ stroke: gradeInfo?.color }}
                  />
                </svg>
                <div className="gauge-text-wrap">
                  <span className="gauge-score-val">{scoreResult.score}</span>
                  <span className="gauge-score-label">SKOR AKHIR</span>
                </div>
              </div>

              <div className="grade-badge-wrap" style={{ background: `${gradeInfo?.color}15`, color: gradeInfo?.color }}>
                <span>Predikat:</span>
                <strong>{gradeInfo?.grade}</strong>
              </div>
            </div>

            {/* Right Result Side: Summary & Details */}
            <div className="score-details-side">
              <h3 className="details-title">Hasil Evaluasi</h3>
              <p className="grade-feedback-text">{gradeInfo?.text}</p>

              {/* Warnings if is_short or is_blank */}
              {scoreResult.features.is_short && (
                <div className="warning-banner">
                  <span>⚠️</span>
                  <span><strong>Perhatian:</strong> Jawaban siswa terindikasi terlalu pendek dari referensi, hal ini dapat menurunkan kualitas penilaian.</span>
                </div>
              )}

              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-indicator-dot" style={{ background: "var(--primary)" }} />
                    <span className="metric-name">Kemiripan Semantik (SBERT)</span>
                  </div>
                  <div className="metric-val">{scoreResult.features.cosine_similarity}%</div>
                  <div className="metric-desc">Kesesuaian makna dan pemahaman konsep secara keseluruhan.</div>
                </div>

                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-indicator-dot" style={{ background: "var(--success)" }} />
                    <span className="metric-name">Kesesuaian Kata Kunci</span>
                  </div>
                  <div className="metric-val">{scoreResult.features.keyword_overlap}%</div>
                  <div className="metric-desc">Persentase kosakata kunci (kata bermakna) yang sepadan.</div>
                </div>

                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-indicator-dot" style={{ background: "var(--info)" }} />
                    <span className="metric-name">Kesesuaian Struktur Kalimat</span>
                  </div>
                  <div className="metric-val">{scoreResult.features.sequence_similarity}%</div>
                  <div className="metric-desc">Tingkat kecocokan sekuensial dan struktur penulisan kalimat.</div>
                </div>

                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-indicator-dot" style={{ background: "var(--text-dim)" }} />
                    <span className="metric-name">Rasio Panjang Jawaban</span>
                  </div>
                  <div className="metric-val">{scoreResult.features.length_ratio}%</div>
                  <div className="metric-desc">Rasio perbandingan jumlah kata siswa ({scoreResult.features.word_count_student} kata) vs kunci ({scoreResult.features.word_count_reference} kata).</div>
                </div>
              </div>

              <div className="result-actions">
                <button className="btn-reset-scoring" onClick={handleReset}>
                  <ArrowLeft size={16} />
                  <span>Nilai Jawaban Lain</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
