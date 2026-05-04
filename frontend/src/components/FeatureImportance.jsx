/**
 * src/components/FeatureImportance.jsx
 * Horizontal bar chart showing per-feature SHAP importances
 * for the current prediction, using Chart.js.
 */

import React, { useMemo } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Title, Tooltip
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { Layers } from 'lucide-react';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip);

// Colours for positive vs negative SHAP influence
const POS_COLOR = '#22c55e';
const NEG_COLOR = '#ef4444';

export default function FeatureImportance({ shapValues }) {
  // shapValues: array of { feature, label, shap_value, input_value, direction }
  // Already sorted by abs(shap_value) DESC from backend

  const chartData = useMemo(() => {
    if (!shapValues || shapValues.length === 0) return null;

    const labels = shapValues.map(f => f.label);
    const values = shapValues.map(f => f.shap_value);
    const colors = values.map(v => v >= 0 ? `${POS_COLOR}cc` : `${NEG_COLOR}cc`);
    const borders = values.map(v => v >= 0 ? POS_COLOR : NEG_COLOR);

    return {
      labels,
      datasets: [{
        label: 'SHAP Value',
        data: values,
        backgroundColor: colors,
        borderColor:     borders,
        borderWidth:     1.5,
        borderRadius:    6,
        borderSkipped:   false,
      }],
    };
  }, [shapValues]);

  const chartOptions = {
    indexAxis: 'y',          // horizontal bars
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0d1f13',
        borderColor:     '#1a3826',
        borderWidth:     1,
        titleColor:      '#86efac',
        bodyColor:       '#94a3b8',
        callbacks: {
          title: ctx => ctx[0].label,
          label: ctx => {
            const val = ctx.raw;
            const dir = val >= 0 ? '▲ Increases' : '▼ Decreases';
            return ` ${dir} prediction by ${Math.abs(val).toFixed(4)}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid:  { color: '#1a382640' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
        title: {
          display: true,
          text:    'SHAP Value (impact on prediction)',
          color:   '#4a7c5e',
          font:    { size: 10 },
        },
      },
      y: {
        grid:  { display: false },
        ticks: { color: '#94a3b8', font: { size: 11 } },
      },
    },
  };

  if (!chartData) return null;

  // Max absolute value for percentage bars in table
  const maxAbs = Math.max(...shapValues.map(f => Math.abs(f.shap_value)));

  return (
    <div className="card p-6 animate-fade-in" id="feature-importance-panel">
      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 mb-5">
        <Layers size={16} style={{ color: '#22c55e' }} />
        <h3 className="section-title mb-0">Feature Importance (SHAP)</h3>
        <div className="ml-auto flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: POS_COLOR }} />
            <span style={{ color: '#86efac' }}>Positive influence</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: NEG_COLOR }} />
            <span style={{ color: '#fca5a5' }}>Negative influence</span>
          </span>
        </div>
      </div>

      {/* ── Chart ──────────────────────────────────────────────── */}
      <div style={{ height: `${Math.max(200, shapValues.length * 38)}px` }}>
        <Bar data={chartData} options={chartOptions} />
      </div>

      {/* ── Feature table with mini bars ─────────────────────── */}
      <div className="mt-5 space-y-2">
        {shapValues.map(({ feature, label, shap_value, input_value, direction }) => {
          const pct = maxAbs > 0 ? (Math.abs(shap_value) / maxAbs) * 100 : 0;
          const color = direction === 'positive' ? POS_COLOR : NEG_COLOR;

          return (
            <div key={feature} className="flex items-center gap-3" id={`shap-row-${feature}`}>
              {/* Feature name + value */}
              <div className="w-36 shrink-0">
                <p className="text-xs font-medium" style={{ color: '#e2e8f0' }}>{label}</p>
                <p className="text-[10px]" style={{ color: '#4a7c5e' }}>
                  Input: {input_value}
                </p>
              </div>

              {/* Mini bar */}
              <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: '#0a1a0f' }}>
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${pct}%`, background: color }}
                />
              </div>

              {/* SHAP value */}
              <div className="w-16 text-right">
                <span className="text-xs font-semibold" style={{ color }}>
                  {shap_value >= 0 ? '+' : ''}{shap_value.toFixed(4)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
