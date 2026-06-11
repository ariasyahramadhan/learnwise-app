import { useState } from "react";
import { Edit, Clipboard, Sliders, ArrowLeft, Trash, CloudUpload } from "../components/Icons";
import "./EssayScoringPage.css";

export default function EssayScoringPage() {
  const [question, setQuestion] = useState("");
  const [referenceAnswer, setReferenceAnswer] = useState("");
  const [studentAnswers, setStudentAnswers] = useState([""]); // Array of student responses
  const [isScoring, setIsScoring] = useState(false);
  const [isParsing, setIsParsing] = useState(false); // File parsing state
  const [showFormatGuide, setShowFormatGuide] = useState(false); // Toggle format instructions
  const [scoreResults, setScoreResults] = useState(null); // Will store list of result objects
  const [expandedIndex, setExpandedIndex] = useState(null); // Track expanded accordion card
  const [error, setError] = useState(null);

  // Add a new empty student answer input field
  const handleAddAnswer = () => {
    setStudentAnswers([...studentAnswers, ""]);
  };

  // Remove a specific student answer input field
  const handleRemoveAnswer = (indexToRemove) => {
    if (studentAnswers.length <= 1) return;
    const updated = studentAnswers.filter((_, idx) => idx !== indexToRemove);
    setStudentAnswers(updated);
  };

  // Update text of a specific student answer
  const handleAnswerChange = (index, value) => {
    const updated = [...studentAnswers];
    updated[index] = value;
    setStudentAnswers(updated);
  };

  // Handle Document Upload and Parsing
  const handleDocUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Check extension
    const allowedExtensions = [".pdf", ".docx", ".txt"];
    const fileExtension = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    
    if (!allowedExtensions.includes(fileExtension)) {
      setError("Format file tidak didukung. Hanya PDF, DOCX, dan TXT.");
      return;
    }

    setIsParsing(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/parse-essay-document", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Gagal mengurai dokumen.");
      }

      const data = await response.json();
      setQuestion(data.question);
      setReferenceAnswer(data.reference_answer);
      setStudentAnswers(data.student_answers);
    } catch (err) {
      setError(err.message || "Gagal mengunggah dan memproses dokumen.");
    } finally {
      setIsParsing(false);
      // Reset input value so same file can be uploaded again
      e.target.value = "";
    }
  };

  const handleScoreEssay = async (e) => {
    e.preventDefault();
    
    // Validations
    if (!referenceAnswer.trim()) {
      setError("Kunci jawaban tidak boleh kosong.");
      return;
    }
    
    // Check if any student answer is empty
    const hasEmpty = studentAnswers.some((ans) => !ans.trim());
    if (hasEmpty) {
      setError("Semua kolom jawaban siswa harus diisi.");
      return;
    }

    setIsScoring(true);
    setError(null);
    setScoreResults(null);
    setExpandedIndex(null);

    try {
      const response = await fetch("http://localhost:8000/api/score-essay", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
          reference_answer: referenceAnswer,
          student_answers: studentAnswers,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Terjadi kesalahan pada server.");
      }

      const data = await response.json();
      setScoreResults(data.results);
      // Auto-expand the first result card
      if (data.results && data.results.length > 0) {
        setExpandedIndex(0);
      }
    } catch (err) {
      setError(err.message || "Gagal menghubungkan ke server.");
    } finally {
      setIsScoring(false);
    }
  };

  const handleReset = () => {
    setQuestion("");
    setReferenceAnswer("");
    setStudentAnswers([""]);
    setScoreResults(null);
    setExpandedIndex(null);
    setError(null);
  };

  const getScoreColor = (score) => {
    if (score >= 85) return "var(--success)";
    if (score >= 70) return "var(--info)";
    if (score >= 55) return "var(--warning)";
    return "var(--danger)";
  };

  const shortPreview = (text, maxLength = 60) => {
    if (!text) return "";
    return text.length > maxLength ? text.slice(0, maxLength) + "..." : text;
  };

  return (
    <div className="essay-scoring-page">
      {/* Header */}
      <div className="scoring-header">
        <h2 className="scoring-title">
          <Edit size={24} style={{ color: "var(--primary)" }} />
          <span>Penilaian Esai Otomatis (Essay Scoring)</span>
        </h2>
        <p className="scoring-subtitle">
          Bandingkan satu atau beberapa jawaban siswa secara semantik dan leksikal dengan kunci jawaban.
        </p>
      </div>

      {error && (
        <div className="error-box">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {!scoreResults ? (
        <form className="scoring-form" onSubmit={handleScoreEssay}>
          
          {/* File Upload Actions Bar */}
          <div className="upload-guide-section">
            <div className="upload-actions-bar">
              <label className={`btn-upload-doc ${isParsing ? "loading" : ""}`}>
                <CloudUpload size={16} />
                <span>{isParsing ? "Sedang Mengurai Dokumen..." : "Upload Dokumen Soal & Jawaban"}</span>
                <input
                  type="file"
                  accept=".pdf,.docx,.txt"
                  onChange={handleDocUpload}
                  style={{ display: "none" }}
                  disabled={isParsing || isScoring}
                />
              </label>
              <button
                type="button"
                className="btn-toggle-guide"
                onClick={() => setShowFormatGuide(!showFormatGuide)}
              >
                {showFormatGuide ? "Sembunyikan Petunjuk Format" : "Petunjuk Format Dokumen"}
              </button>
            </div>

            {showFormatGuide && (
              <div className="format-guide-card">
                <h4>💡 Petunjuk Format Dokumen untuk Pemetaan Otomatis</h4>
                <p>Agar teks di dalam file <strong>(.pdf, .docx, .txt)</strong> Anda otomatis dipetakan ke kolom Soal, Kunci Jawaban, dan Daftar Jawaban Siswa secara tepat, susunlah tulisan dokumen Anda menggunakan label penanda berikut:</p>
                <pre className="format-example-code">
{`Soal: [Masukkan teks soal esai di sini]

Kunci Jawaban: [Masukkan teks kunci jawaban di sini]

Jawaban Siswa 1: [Jawaban dari siswa pertama]

Jawaban Siswa 2: [Jawaban dari siswa kedua]

Jawaban Siswa 3: [Jawaban dari siswa ketiga]`}
                </pre>
                <p className="format-guide-note">
                  * Catatan: Sistem memindai pola kata secara fleksibel, sehingga Anda bisa menulis <code>Soal:</code>, <code>Kunci Jawaban:</code>, atau <code>Jawaban 1:</code> tanpa mempermasalahkan huruf besar/kecil.
                </p>
              </div>
            )}
          </div>

          {/* Staged Inputs Stack (Vertical Full-Width Layout) */}
          <div className="scoring-inputs-stack">
            
            {/* 1. Soal / Pertanyaan */}
            <div className="input-card question-card">
              <label className="input-label">
                <Clipboard size={16} style={{ color: "var(--primary)" }} />
                <span>Soal / Pertanyaan (Pertanyaan Esai - Opsional)</span>
              </label>
              <textarea
                className="scoring-textarea question-textarea"
                placeholder="Masukkan teks soal atau petunjuk esai di sini sebagai konteks..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                maxLength={2000}
                disabled={isScoring}
                style={{ minHeight: "80px" }}
              />
            </div>

            {/* 2. Kunci Jawaban */}
            <div className="input-card reference-card">
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
                style={{ minHeight: "150px" }}
              />
              <div className="char-counter">
                {referenceAnswer.length} / 5000 karakter
              </div>
            </div>

            {/* 3. Multiple Student Answers */}
            <div className="answers-stack-container">
              <div className="answers-stack-header">
                <label className="input-label">
                  <Edit size={16} style={{ color: "var(--primary)" }} />
                  <span>Daftar Jawaban Siswa</span>
                </label>
                <span className="answers-count-badge">
                  {studentAnswers.length} Jawaban
                </span>
              </div>

              <div className="answers-scroll-area-full">
                {studentAnswers.map((answer, index) => (
                  <div key={index} className="input-card student-answer-card">
                    <div className="student-card-header">
                      <span className="student-label-text">Jawaban Siswa #{index + 1}</span>
                      {studentAnswers.length > 1 && (
                        <button
                          type="button"
                          className="btn-remove-answer"
                          onClick={() => handleRemoveAnswer(index)}
                          title="Hapus jawaban ini"
                          disabled={isScoring}
                        >
                          <Trash size={14} />
                        </button>
                      )}
                    </div>
                    <textarea
                      className="scoring-textarea student-textarea"
                      placeholder={`Masukkan teks jawaban dari siswa #${index + 1} di sini...`}
                      value={answer}
                      onChange={(e) => handleAnswerChange(index, e.target.value)}
                      maxLength={5000}
                      disabled={isScoring}
                      style={{ minHeight: "100px" }}
                    />
                    <div className="char-counter">
                      {answer.length} / 5000 karakter
                    </div>
                  </div>
                ))}
              </div>

              <button
                type="button"
                className="btn-add-answer-field"
                onClick={handleAddAnswer}
                disabled={isScoring}
              >
                <span>+</span> Tambah Jawaban Siswa Lain
              </button>
            </div>
          </div>

          <button
            type="submit"
            className={`btn-score ${isScoring ? "loading" : ""}`}
            disabled={isScoring || isParsing || !referenceAnswer.trim() || studentAnswers.some((ans) => !ans.trim())}
          >
            {isScoring ? (
              <>
                <div className="spinner" />
                <span>Sedang Menilai Jawaban ({studentAnswers.length} esai)...</span>
              </>
            ) : (
              <>
                <Sliders size={20} />
                <span>Mulai Penilaian ({studentAnswers.length} esai)</span>
              </>
            )}
          </button>
        </form>
      ) : (
        <div className="scoring-results-panel">
          <div className="results-panel-header">
            <div>
              <h3 className="results-panel-title">Hasil Penilaian Esai</h3>
              <p className="results-panel-desc">
                Berhasil menilai {scoreResults.length} jawaban siswa berdasarkan kunci jawaban.
              </p>
            </div>
            <button className="btn-reset-scoring" onClick={handleReset}>
              <ArrowLeft size={14} />
              <span>Buat Penilaian Baru</span>
            </button>
          </div>

          {/* Vertical Accordion Stack */}
          <div className="accordion-stack">
            {scoreResults.map((result, index) => {
              const isExpanded = expandedIndex === index;
              const themeColor = getScoreColor(result.score);
              const answerText = studentAnswers[index];
              
              return (
                <div
                  key={index}
                  className={`accordion-item ${isExpanded ? "expanded" : ""}`}
                  style={{ borderLeft: `4px solid ${themeColor}` }}
                >
                  {/* Accordion Header */}
                  <div
                    className="accordion-header"
                    onClick={() => setExpandedIndex(isExpanded ? null : index)}
                  >
                    <div className="accordion-header-left">
                      <span className="student-badge" style={{ background: `${themeColor}15`, color: themeColor }}>
                        Siswa #{index + 1}
                      </span>
                      <span className="answer-text-preview">
                        {shortPreview(answerText)}
                      </span>
                    </div>

                    <div className="accordion-header-right">
                      <div className="score-badge" style={{ color: themeColor }}>
                        <span className="score-num-label">Nilai:</span>
                        <strong className="score-num-val">{result.score}</strong>
                      </div>
                      <span className="chevron-icon">{isExpanded ? "▲" : "▼"}</span>
                    </div>
                  </div>

                  {/* Accordion Body (Expanded view) */}
                  {isExpanded && (
                    <div className="accordion-body">
                      <div className="expanded-result-grid">
                        {/* Gauge container */}
                        <div className="expanded-gauge-column">
                          <div className="gauge-outer-small">
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
                                strokeDashoffset={251.2 - (251.2 * result.score) / 100}
                                style={{ stroke: themeColor }}
                              />
                            </svg>
                            <div className="gauge-text-wrap-small">
                              <span className="gauge-score-val-small">{result.score}</span>
                              <span className="gauge-score-label-small">NILAI</span>
                            </div>
                          </div>
                        </div>

                        {/* Details Grid */}
                        <div className="expanded-details-column">
                          {/* Student Answer Paragraph Preview */}
                          <div className="student-full-text-preview">
                            <strong>Jawaban yang dinilai:</strong>
                            <p>{answerText}</p>
                          </div>

                          {/* Warnings */}
                          {result.features.is_short && (
                            <div className="warning-banner" style={{ marginBottom: "1rem" }}>
                              <span>⚠️</span>
                              <span><strong>Peringatan:</strong> Jawaban siswa terlalu pendek. Rasio kata hanya {result.features.length_ratio}% dibandingkan kunci jawaban.</span>
                            </div>
                          )}

                          <div className="metrics-grid">
                            <div className="metric-card">
                              <div className="metric-header">
                                <span className="metric-indicator-dot" style={{ background: "var(--primary)" }} />
                                <span className="metric-name">Kemiripan Semantik (SBERT)</span>
                              </div>
                              <div className="metric-val">{result.features.cosine_similarity}%</div>
                              <div className="metric-desc">Kesesuaian makna dan penyampaian konsep.</div>
                            </div>

                            <div className="metric-card">
                              <div className="metric-header">
                                <span className="metric-indicator-dot" style={{ background: "var(--success)" }} />
                                <span className="metric-name">Kesesuaian Kata Kunci</span>
                              </div>
                              <div className="metric-val">{result.features.keyword_overlap}%</div>
                              <div className="metric-desc">Kesesuaian kosakata penting (non-stopwords).</div>
                            </div>

                            <div className="metric-card">
                              <div className="metric-header">
                                <span className="metric-indicator-dot" style={{ background: "var(--info)" }} />
                                <span className="metric-name">Struktur Kalimat</span>
                              </div>
                              <div className="metric-val">{result.features.sequence_similarity}%</div>
                              <div className="metric-desc">Kecocokan sekuensial penulisan teks.</div>
                            </div>

                            <div className="metric-card">
                              <div className="metric-header">
                                <span className="metric-indicator-dot" style={{ background: "var(--text-dim)" }} />
                                <span className="metric-name">Panjang Jawaban</span>
                              </div>
                              <div className="metric-val">{result.features.length_ratio}%</div>
                              <div className="metric-desc">Jumlah kata: {result.features.word_count_student} (kunci: {result.features.word_count_reference}).</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
