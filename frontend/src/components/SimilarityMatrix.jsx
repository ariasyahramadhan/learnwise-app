import "./SimilarityMatrix.css";

function getColor(value) {
  if (value === 100) return "rgba(249,115,22,0.12)"; // diagonal
  if (value >= 80) return "rgba(239,68,68,0.55)";
  if (value >= 60) return "rgba(245,158,11,0.45)";
  if (value >= 40) return "rgba(249,115,22,0.25)";
  if (value >= 20) return "rgba(16,185,129,0.25)";
  return "rgba(16,185,129,0.1)";
}

function shortName(name) {
  return name.length > 12 ? name.slice(0, 10) + "…" : name;
}

export default function SimilarityMatrix({ matrix, documents, onCompare }) {
  return (
    <div className="matrix-scroll">
      <table className="matrix-table">
        <thead>
          <tr>
            <th className="matrix-corner" />
            {documents.map((doc, j) => (
              <th key={j} className="matrix-col-header">
                <div style={{ display: "flex", flexDirection: "column", gap: "2px", alignItems: "center" }}>
                  <span>#{j + 1}</span>
                  <span className="col-name">{shortName(doc.name)}</span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="matrix-row-header">
                <div style={{ display: "flex", gap: "0.35rem", alignItems: "center", justifyContent: "flex-end" }}>
                  <span style={{ color: "var(--primary)", fontWeight: 800 }}>#{i + 1}</span>
                  <span>{shortName(documents[i].name)}</span>
                </div>
              </td>
              {row.map((val, j) => (
                <td
                  key={j}
                  className="matrix-cell"
                  style={{
                    background: getColor(i === j ? 100 : val),
                    cursor: i === j ? "default" : "pointer"
                  }}
                  title={i === j ? undefined : `${documents[i].name} vs ${documents[j].name}: ${val}%`}
                  onClick={() => {
                    if (i !== j && onCompare) {
                      onCompare({
                        doc_a: documents[i].name,
                        doc_b: documents[j].name,
                        similarity: val,
                        risk: val >= 80 ? "high" : val >= 50 ? "medium" : "low"
                      });
                    }
                  }}
                >
                  <span className="cell-val">
                    {i === j ? "—" : `${val}%`}
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
