import React, { useMemo, useState } from "react";
import { Close } from "./Icons";
import "./CompareModal.css";

// Helper to split text into sentences (handles basic punctuation endings)
function splitIntoSentences(text) {
  if (!text) return [];
  // Match sentences including trailing punctuation
  const matches = text.match(/[^.!?]+[.!?]*/g) || [text];
  return matches.map(s => s.trim()).filter(Boolean);
}

// Jaccard similarity helper on words
function calculateJaccardSimilarity(sentA, sentB) {
  const cleanA = sentA.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean);
  const cleanB = sentB.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean);
  
  if (cleanA.length === 0 || cleanB.length === 0) return 0;
  
  const setA = new Set(cleanA);
  const setB = new Set(cleanB);
  
  const intersection = new Set([...setA].filter(x => setB.has(x)));
  const union = new Set([...setA, ...setB]);
  
  return intersection.size / union.size;
}

export default function CompareModal({ pair, onClose, documents }) {
  const [hoveredPairId, setHoveredPairId] = useState(null);

  const docA = useMemo(() => documents.find(d => d.name === pair.doc_a), [documents, pair]);
  const docB = useMemo(() => documents.find(d => d.name === pair.doc_b), [documents, pair]);

  const textA = docA?.text || "";
  const textB = docB?.text || "";

  const sentencesA = useMemo(() => splitIntoSentences(textA), [textA]);
  const sentencesB = useMemo(() => splitIntoSentences(textB), [textB]);

  // Compute sentence matches
  const { matchesA, matchesB, matchCount } = useMemo(() => {
    const matchesA = {};
    const matchesB = {};
    let matchCount = 0;

    sentencesA.forEach((sA, idxA) => {
      let bestScore = 0;
      let bestIdxB = -1;

      sentencesB.forEach((sB, idxB) => {
        const score = calculateJaccardSimilarity(sA, sB);
        if (score > bestScore) {
          bestScore = score;
          bestIdxB = idxB;
        }
      });

      // Threshold 0.35 matches typical plagiarism parafrase
      if (bestScore >= 0.35) {
        const pairId = `${idxA}-${bestIdxB}`;
        matchesA[idxA] = { partner: bestIdxB, pairId };
        
        if (!matchesB[bestIdxB] || matchesB[bestIdxB].score < bestScore) {
          matchesB[bestIdxB] = { partner: idxA, pairId, score: bestScore };
        }
        matchCount++;
      }
    });

    return { matchesA, matchesB, matchCount };
  }, [sentencesA, sentencesB]);

  const handleMouseEnter = (pairId) => {
    setHoveredPairId(pairId);
  };

  const handleMouseLeave = () => {
    setHoveredPairId(null);
  };

  return (
    <div className="compare-overlay" onClick={onClose}>
      <div className="compare-container" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="compare-header">
          <div className="compare-header-info">
            <h3 className="compare-title">Bandingkan Dokumen</h3>
            <p className="compare-subtitle">
              Menampilkan pasangan kalimat yang mirip berdasarkan leksikal gaya penulisan.
            </p>
          </div>
          <div className="compare-stats">
            <div className="stat-pill">
              <span>Kemiripan Pasangan:</span>
              <strong className={`score-text-${pair.risk}`}>{pair.similarity}%</strong>
            </div>
            <div className="stat-pill">
              <span>Kalimat Mirip:</span>
              <strong>{matchCount} kalimat</strong>
            </div>
            <button className="btn-close-compare" onClick={onClose}>
              <Close size={20} />
            </button>
          </div>
        </div>

        {/* Panes */}
        <div className="compare-panes">
          {/* Doc A */}
          <div className="compare-pane">
            <div className="pane-header">
              <span className="pane-badge badge-a">Dokumen A</span>
              <span className="pane-filename">{pair.doc_a}</span>
            </div>
            <div className="pane-content">
              {sentencesA.map((sentence, i) => {
                const match = matchesA[i];
                if (match) {
                  const isActive = match.pairId === hoveredPairId;
                  return (
                    <span
                      key={i}
                      className={`sentence-highlight ${isActive ? "active" : ""}`}
                      onMouseEnter={() => handleMouseEnter(match.pairId)}
                      onMouseLeave={handleMouseLeave}
                    >
                      {sentence}{" "}
                    </span>
                  );
                }
                return <span key={i} className="sentence-text">{sentence} </span>;
              })}
            </div>
          </div>

          {/* Doc B */}
          <div className="compare-pane">
            <div className="pane-header">
              <span className="pane-badge badge-b">Dokumen B</span>
              <span className="pane-filename">{pair.doc_b}</span>
            </div>
            <div className="pane-content">
              {sentencesB.map((sentence, i) => {
                const match = matchesB[i];
                if (match) {
                  const isActive = match.pairId === hoveredPairId;
                  return (
                    <span
                      key={i}
                      className={`sentence-highlight ${isActive ? "active" : ""}`}
                      onMouseEnter={() => handleMouseEnter(match.pairId)}
                      onMouseLeave={handleMouseLeave}
                    >
                      {sentence}{" "}
                    </span>
                  );
                }
                return <span key={i} className="sentence-text">{sentence} </span>;
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
