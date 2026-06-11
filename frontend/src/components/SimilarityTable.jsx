import "./SimilarityTable.css";

export default function SimilarityTable({ pairs, onCompare }) {
  return (
    <div className="table-wrapper">
      <table className="sim-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Dokumen A</th>
            <th>Dokumen B</th>
            <th>Kemiripan</th>
            <th>Status</th>
            <th style={{ textAlign: "right" }}>Aksi</th>
          </tr>
        </thead>
        <tbody>
          {pairs.map((pair, i) => (
            <tr key={i} className={`row-${pair.risk}`}>
              <td className="col-num">{i + 1}</td>
              <td className="col-name">{pair.doc_a}</td>
              <td className="col-name">{pair.doc_b}</td>
              <td className="col-score">
                <div className="score-bar-wrap">
                  <div
                    className={`score-bar bar-${pair.risk}`}
                    style={{ width: `${pair.similarity}%` }}
                  />
                  <span className={`score-text text-${pair.risk}`}>
                    {pair.similarity}%
                  </span>
                </div>
              </td>
              <td>
                <span className={`badge badge-${pair.risk}`}>
                  {pair.risk === "high" ? "Tinggi" : pair.risk === "medium" ? "Sedang" : "Rendah"}
                </span>
              </td>
              <td style={{ textAlign: "right" }}>
                {onCompare && (
                  <button
                    className="btn-compare-action"
                    onClick={() => onCompare(pair)}
                    style={{
                      padding: "0.35rem 0.75rem",
                      fontSize: "0.72rem",
                      fontWeight: 700,
                      background: "var(--primary-subtle)",
                      color: "var(--primary-dark)",
                      border: "none",
                      borderRadius: "6px",
                      cursor: "pointer",
                      transition: "all 0.15s"
                    }}
                    onMouseEnter={(e) => {
                      e.target.style.background = "var(--primary)";
                      e.target.style.color = "white";
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.background = "var(--primary-subtle)";
                      e.target.style.color = "var(--primary-dark)";
                    }}
                  >
                    Bandingkan
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
