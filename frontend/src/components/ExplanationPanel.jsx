/**
 * src/components/ExplanationPanel.jsx
 * Shows the human-readable SHAP text explanation
 * and optionally renders the backend-generated SHAP plot image.
 */

import React, { useState } from 'react';
import { Lightbulb, Image, ChevronDown, ChevronUp, Brain } from 'lucide-react';
import FeatureImportance from './FeatureImportance';

export default function ExplanationPanel({ explanation }) {
  const [showPlot, setShowPlot] = useState(false);

  if (!explanation) return null;

  const {
    crop,
    shap_values,
    text_explanation,
    summary_plot_base64,
    top_features,
  } = explanation;

  // Parse markdown bold (**text**) to JSX
  const parseExplanation = (text) => {
    const parts = text.split(/\*\*(.*?)\*\*/g);
    return parts.map((part, i) =>
      i % 2 === 1
        ? <strong key={i} style={{ color: '#22c55e' }}>{part}</strong>
        : part
    );
  };

  return (
    <div className="space-y-4 animate-slide-up" id="explanation-panel">

      {/* ── XAI Text Explanation Card ──────────────────────────── */}
      <div className="card p-6" id="text-explanation-card">
        <div className="flex items-center gap-2 mb-4">
          <div className="p-2 rounded-xl" style={{ background: '#f59e0b1a', border: '1px solid #f59e0b33' }}>
            <Lightbulb size={15} style={{ color: '#fbbf24' }} />
          </div>
          <div>
            <h3 className="font-semibold text-sm" style={{ color: '#fbbf24' }}>
              Why was this recommended?
            </h3>
            <p className="text-[10px]" style={{ color: '#4a7c5e' }}>
              SHAP Explainability — based on Shapley values
            </p>
          </div>
          <div className="ml-auto flex items-center gap-1 badge badge-yellow">
            <Brain size={10} />
            XAI
          </div>
        </div>

        {/* Explanation text */}
        <p
          className="text-sm leading-relaxed px-4 py-3 rounded-xl"
          style={{ background: '#0a1a0f', border: '1px solid #1a3826', color: '#cbd5e1' }}
        >
          {parseExplanation(text_explanation)}
        </p>

        {/* Top 3 influencing factors */}
        <div className="mt-4 grid grid-cols-3 gap-2">
          {top_features.slice(0, 3).map((f, i) => (
            <div
              key={f.feature}
              className="rounded-xl p-3 text-center"
              style={{ background: '#0a1a0f', border: '1px solid #1a3826' }}
            >
              <div className="text-[10px] mb-1 font-semibold uppercase tracking-widest" style={{ color: '#4a7c5e' }}>
                #{i + 1} Factor
              </div>
              <div className="text-xs font-semibold" style={{ color: '#86efac' }}>
                {f.feature}
              </div>
              <div className="text-[11px] mt-0.5" style={{ color: '#22c55e' }}>
                {f.importance.toFixed(4)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── SHAP Feature Importance Chart ─────────────────────── */}
      {shap_values && shap_values.length > 0 && (
        <FeatureImportance shapValues={shap_values} />
      )}

      {/* ── Backend SHAP Plot (collapsible) ───────────────────── */}
      {summary_plot_base64 && (
        <div className="card overflow-hidden" id="shap-plot-card">
          <button
            className="w-full flex items-center justify-between px-5 py-4 text-sm font-medium transition-all"
            style={{ color: '#86efac' }}
            onClick={() => setShowPlot(!showPlot)}
            id="toggle-shap-plot"
          >
            <span className="flex items-center gap-2">
              <Image size={14} />
              SHAP Summary Plot (Matplotlib)
            </span>
            {showPlot ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {showPlot && (
            <div className="px-5 pb-5 animate-fade-in">
              <img
                src={`data:image/png;base64,${summary_plot_base64}`}
                alt="SHAP Summary Plot"
                className="w-full rounded-xl"
                style={{ border: '1px solid #1a3826' }}
              />
              <p className="text-[10px] mt-2 text-center" style={{ color: '#4a7c5e' }}>
                Green = pushed towards prediction · Red = pushed away from prediction
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
