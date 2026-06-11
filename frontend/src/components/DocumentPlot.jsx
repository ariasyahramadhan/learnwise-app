import { useState, useMemo } from "react";
import "./DocumentPlot.css";

const COLORS = [
  "#F97316", "#3B82F6", "#10B981", "#8B5CF6",
  "#EF4444", "#06B6D4", "#F59E0B", "#EC4899",
  "#14B8A6", "#6366F1"
];

const W = 600, H = 400, PAD = 60;

function toSVG(coord, w, h, pad) {
  return {
    x: pad + ((coord.x + 1) / 2) * (w - pad * 2),
    y: (h - pad) - ((coord.y + 1) / 2) * (h - pad * 2)
  };
}

export default function DocumentPlot({ documents, coordinates, pairs }) {
  const [hovered, setHovered] = useState(null);
  const [tooltip, setTooltip] = useState(null);

  const points = useMemo(() =>
    documents.map((doc, i) => ({
      ...doc,
      color: COLORS[i % COLORS.length],
      ...toSVG(coordinates[i], W, H, PAD)
    })), [documents, coordinates]);

  const edges = useMemo(() =>
    pairs.map((pair, i) => {
      const a = points[pair.doc_a_index];
      const b = points[pair.doc_b_index];
      const opacity = (pair.similarity / 100) * 0.65;
      let stroke = "#10B981";
      if (pair.risk === "high") stroke = "#EF4444";
      else if (pair.risk === "medium") stroke = "#F59E0B";
      return { ...pair, x1: a.x, y1: a.y, x2: b.x, y2: b.y, opacity, stroke, key: i };
    }), [pairs, points]);

  const handleMouseEnter = (point) => {
    setHovered(point.index);
    setTooltip({ ...point });
  };
  const handleMouseLeave = () => {
    setHovered(null);
    setTooltip(null);
  };

  return (
    <div className="plot-wrapper">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="document-plot"
        style={{ width: "100%", height: "auto" }}
      >
        <line x1={PAD} y1={H / 2} x2={W - PAD} y2={H / 2} stroke="#E5E7EB" strokeWidth="1" strokeDasharray="4" />
        <line x1={W / 2} y1={PAD} x2={W / 2} y2={H - PAD} stroke="#E5E7EB" strokeWidth="1" strokeDasharray="4" />

        {edges.map(e => (
          <line
            key={e.key}
            x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
            stroke={e.stroke}
            strokeWidth={hovered === e.doc_a_index || hovered === e.doc_b_index ? 2.5 : 1.5}
            strokeOpacity={hovered !== null
              ? (hovered === e.doc_a_index || hovered === e.doc_b_index ? 0.9 : 0.08)
              : e.opacity}
            strokeDasharray={e.risk === "low" ? "5 5" : "none"}
          />
        ))}

        {hovered !== null && edges
          .filter(e => e.doc_a_index === hovered || e.doc_b_index === hovered)
          .map(e => (
            <text
              key={`label-${e.key}`}
              x={(e.x1 + e.x2) / 2}
              y={(e.y1 + e.y2) / 2 - 7}
              textAnchor="middle"
              fontSize="11"
              fill={e.stroke}
              fontFamily="'Fira Code', monospace"
              fontWeight="700"
            >
              {e.similarity}%
            </text>
          ))
        }

        {points.map((p) => (
          <g
            key={p.index}
            onMouseEnter={() => handleMouseEnter(p)}
            onMouseLeave={handleMouseLeave}
            style={{ cursor: "pointer" }}
          >
            <circle
              cx={p.x} cy={p.y}
              r={hovered === p.index ? 17 : 13}
              fill={p.color}
              fillOpacity={hovered === null || hovered === p.index ? 1 : 0.2}
              style={{ transition: "r 0.15s, fill-opacity 0.15s", filter: hovered === p.index ? `drop-shadow(0 3px 8px ${p.color}88)` : "none" }}
            />
            <text
              x={p.x} y={p.y + 1}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="11"
              fontWeight="800"
              fill="white"
              fontFamily="'Plus Jakarta Sans', sans-serif"
              style={{ pointerEvents: "none" }}
            >
              {p.index + 1}
            </text>
            <text
              x={p.x} y={p.y + 24}
              textAnchor="middle"
              fontSize="10"
              fill={p.color}
              fontFamily="'Plus Jakarta Sans', sans-serif"
              fontWeight="700"
              style={{ pointerEvents: "none" }}
            >
              {p.name.length > 14 ? p.name.slice(0, 12) + "…" : p.name}
            </text>
          </g>
        ))}
      </svg>

      <div className="plot-legend">
        <span className="legend-item"><span className="legend-line high" /> Kemiripan Tinggi (≥80%)</span>
        <span className="legend-item"><span className="legend-line medium" /> Sedang (50–79%)</span>
        <span className="legend-item"><span className="legend-line low" /> Rendah (&lt;50%)</span>
      </div>

      {tooltip && (
        <div className="plot-tooltip">
          <strong>#{tooltip.index + 1} {tooltip.name}</strong>
          <span>{tooltip.word_count?.toLocaleString()} kata</span>
        </div>
      )}
    </div>
  );
}
