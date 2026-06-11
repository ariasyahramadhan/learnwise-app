import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";
import EssayScoringPage from "./pages/EssayScoringPage";
import { Search, Edit, Clipboard, GraduationCap, ArrowLeft } from "./components/Icons";
import "./App.css";

const TABS = [
  { id: "plagiarism", label: "Deteksi Plagiarisme", icon: Search },
  { id: "essay_scoring", label: "Scoring Essay", icon: Edit },
  { id: "summarize", label: "Pembeda Teks Antara AI / Manusia", icon: Clipboard },
];

export default function App() {
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState("plagiarism");

  const handleResult = (data) => setResult(data);
  const handleReset = () => setResult(null);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">
              <GraduationCap size={20} />
            </div>
            <span className="logo-text">Learn<span>Wise</span></span>
          </div>

          <nav className="header-nav">
            {TABS.map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  className={`nav-link ${activeTab === tab.id ? "active" : ""}`}
                  onClick={() => { setActiveTab(tab.id); setResult(null); }}
                  style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
                >
                  <Icon size={16} />
                  {tab.label}
                </button>
              );
            })}
          </nav>

          <div className="header-actions">
            {result && (
              <button className="btn-reset" onClick={handleReset} style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                <ArrowLeft size={16} />
                <span>Analisis Baru</span>
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="app-main">
        {activeTab === "plagiarism" ? (
          !result ? (
            <UploadPage
              onResult={handleResult}
              isAnalyzing={isAnalyzing}
              setIsAnalyzing={setIsAnalyzing}
            />
          ) : (
            <ResultPage result={result} onReset={handleReset} />
          )
        ) : activeTab === "essay_scoring" ? (
          <EssayScoringPage />
        ) : (
          <ComingSoon tab={TABS.find(t => t.id === activeTab)} />
        )}
      </main>
    </div>
  );
}

function ComingSoon({ tab }) {
  return (
    <div style={{
      maxWidth: 500,
      margin: "6rem auto",
      textAlign: "center",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      gap: "1rem"
    }}>
      <div style={{ fontSize: "3.5rem" }}>{tab?.icon}</div>
      <h2 style={{ fontSize: "1.5rem", fontWeight: 700 }}>{tab?.label}</h2>
      <p style={{ color: "var(--text-dim)", lineHeight: 1.6 }}>
        Fitur ini sedang dalam pengembangan. Nantikan pembaruan LearnWise berikutnya!
      </p>
      <div style={{
        marginTop: "0.5rem",
        padding: "0.4rem 1rem",
        background: "var(--primary-subtle)",
        color: "var(--primary-dark)",
        borderRadius: 99,
        fontSize: "0.8rem",
        fontWeight: 600
      }}>Segera Hadir</div>
    </div>
  );
}
