/**
 * src/components/ResultCard.jsx
 * Shows the primary crop recommendation with confidence and crop emoji.
 */

import React from 'react';
import { CheckCircle, TrendingUp, AlertTriangle } from 'lucide-react';

// Crop emoji map for visual flair
const CROP_EMOJI = {
  rice: '🌾', maize: '🌽', chickpea: '🫘', kidneybeans: '🫘',
  pigeonpeas: '🌿', mothbeans: '🫘', mungbean: '🫛', blackgram: '🫘',
  lentil: '🫘', pomegranate: '🍎', banana: '🍌', mango: '🥭',
  grapes: '🍇', watermelon: '🍉', muskmelon: '🍈', apple: '🍎',
  orange: '🍊', papaya: '🧡', coconut: '🥥', cotton: '🌸',
  jute: '🌿', coffee: '☕',
};

// Confidence colour thresholds
const getConfidenceStyle = (conf) => {
  if (conf >= 0.85) return { color: '#22c55e', label: 'Very High', badge: 'badge-green' };
  if (conf >= 0.70) return { color: '#84cc16', label: 'High',      badge: 'badge-green' };
  if (conf >= 0.55) return { color: '#f59e0b', label: 'Medium',    badge: 'badge-yellow' };
  return              { color: '#ef4444', label: 'Low',       badge: 'badge-red' };
};

export default function ResultCard({ result }) {
  if (!result) return null;

  const { crop, confidence, model_used } = result.primary_prediction ?? {};

  // The backend now returns 503 rather than a placeholder crop, but never
  // render an unusable value as if it were a recommendation: a fabricated
  // "unknown" at 0% confidence used to reach users looking entirely normal.
  const isUsable =
    typeof crop === 'string' &&
    crop.length > 0 &&
    crop !== 'unknown' &&
    Number.isFinite(confidence) &&
    confidence > 0;

  if (!isUsable) {
    return (
      <div
        className="card p-6 animate-slide-up"
        id="result-card-unavailable"
        role="alert"
        style={{ borderColor: '#f59e0b44' }}
      >
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle size={16} style={{ color: '#f59e0b' }} />
          <span
            className="text-xs font-semibold uppercase tracking-widest"
            style={{ color: '#f59e0b' }}
          >
            No recommendation available
          </span>
        </div>
        <p className="text-sm" style={{ color: '#cbd5e1' }}>
          The prediction service returned an incomplete result, so no crop can be
          recommended for these conditions.
        </p>
        <p className="text-xs mt-2" style={{ color: '#4a7c5e' }}>
          If you are running this locally, train the models first:{' '}
          <code style={{ color: '#86efac' }}>python train_pipeline.py</code>
        </p>
      </div>
    );
  }

  const emoji  = CROP_EMOJI[crop] || '🌱';
  const pct    = Math.round(confidence * 100);
  const cs     = getConfidenceStyle(confidence);

  return (
    <div className="card-glow p-6 animate-slide-up" id="result-card">
      {/* ── Header badge ─────────────────────────────────── */}
      <div className="flex items-center gap-2 mb-4">
        <CheckCircle size={16} style={{ color: '#22c55e' }} />
        <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: '#22c55e' }}>
          Recommended Crop
        </span>
        <span className={`badge ${cs.badge} ml-auto`}>
          {cs.label} Confidence
        </span>
      </div>

      {/* ── Main crop display ─────────────────────────────── */}
      <div className="flex items-center gap-4 mb-5">
        <div
          className="text-6xl select-none"
          style={{ filter: 'drop-shadow(0 0 20px rgba(34,197,94,0.4))' }}
          role="img" aria-label={crop}
        >
          {emoji}
        </div>
        <div>
          <h2
            className="text-3xl font-bold capitalize"
            style={{ color: '#86efac' }}
          >
            {crop}
          </h2>
          <p className="text-xs mt-1" style={{ color: '#4a7c5e' }}>
            Model: {model_used}
          </p>
        </div>
      </div>

      {/* ── Confidence meter ──────────────────────────────── */}
      <div className="mb-2">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-medium" style={{ color: '#94a3b8' }}>
            Prediction Confidence
          </span>
          <span className="text-lg font-bold" style={{ color: cs.color }}>
            {pct}%
          </span>
        </div>
        <div className="h-3 rounded-full overflow-hidden" style={{ background: '#1a3020' }}>
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{
              width: `${pct}%`,
              background: `linear-gradient(90deg, ${cs.color}88, ${cs.color})`,
              boxShadow: `0 0 12px ${cs.color}60`,
            }}
          />
        </div>
      </div>

      {/* ── Insight footer ────────────────────────────────── */}
      <div
        className="mt-4 px-4 py-3 rounded-xl flex items-center gap-2"
        style={{ background: '#16a34a12', border: '1px solid #16a34a22' }}
      >
        <TrendingUp size={14} style={{ color: '#4ade80' }} />
        <p className="text-xs" style={{ color: '#86efac' }}>
          <strong>{crop.charAt(0).toUpperCase() + crop.slice(1)}</strong> is the optimal
          crop for these soil and environmental conditions.
        </p>
      </div>
    </div>
  );
}
