/**
 * src/components/ModelComparison.jsx
 * Shows a table + bar chart comparing RF vs LR vs NaiveBayes metrics.
 * Data comes from GET /api/v1/metrics and the predict response.
 */

import React, { useEffect, useState } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Title, Tooltip, Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { BarChart2, Trophy, Cpu } from 'lucide-react';
import { getMetrics } from '../api/client';
import { SkeletonCard } from './LoadingSpinner';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const MODEL_COLORS = {
  RandomForest:       { bar: '#22c55e', bg: '#22c55e22', border: '#22c55e44', label: 'Random Forest' },
  LogisticRegression: { bar: '#3b82f6', bg: '#3b82f622', border: '#3b82f644', label: 'Logistic Reg.' },
  NaiveBayes:         { bar: '#a855f7', bg: '#a855f722', border: '#a855f744', label: 'Naive Bayes'   },
};

const METRIC_LABELS = {
  accuracy:  'Accuracy',
  precision: 'Precision',
  recall:    'Recall',
  f1_score:  'F1 Score',
};

export default function ModelComparison({ allModelsPrediction }) {
  const [metrics,  setMetrics]  = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);
  const [activeMetric, setActiveMetric] = useState('accuracy');

  // Fetch training metrics from backend
  useEffect(() => {
    getMetrics()
      .then(data => { setMetrics(data.models); setLoading(false); })
      .catch(() => {
        setError('Metrics not available — run training pipeline first.');
        setLoading(false);
      });
  }, []);

  if (loading) return <SkeletonCard />;

  if (error) return (
    <div className="card p-6">
      <p className="text-sm" style={{ color: '#ef4444' }}>{error}</p>
    </div>
  );

  const modelNames = Object.keys(metrics);

  // ── Bar chart data ─────────────────────────────────────────────────────────
  const chartData = {
    labels: modelNames.map(n => MODEL_COLORS[n]?.label || n),
    datasets: [{
      label: METRIC_LABELS[activeMetric],
      data: modelNames.map(n => metrics[n][activeMetric] ?? 0),
      backgroundColor: modelNames.map(n => MODEL_COLORS[n]?.bg || '#22c55e22'),
      borderColor:     modelNames.map(n => MODEL_COLORS[n]?.bar || '#22c55e'),
      borderWidth:     2,
      borderRadius:    8,
      borderSkipped:   false,
    }],
  };

  const chartOptions = {
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
          label: ctx => ` ${(ctx.raw * 100).toFixed(2)}%`,
        },
      },
    },
    scales: {
      x: {
        grid:  { color: '#1a382612' },
        ticks: { color: '#94a3b8', font: { size: 11 } },
      },
      y: {
        min:  0.5,
        max:  1.0,
        grid: { color: '#1a382630' },
        ticks: {
          color: '#94a3b8',
          font: { size: 11 },
          callback: v => `${(v * 100).toFixed(0)}%`,
        },
      },
    },
  };

  // Best model by selected metric
  const bestModel = modelNames.reduce((a, b) =>
    (metrics[a][activeMetric] ?? 0) > (metrics[b][activeMetric] ?? 0) ? a : b
  );

  return (
    <div className="card p-6 animate-fade-in" id="model-comparison-panel">
      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 mb-5">
        <BarChart2 size={16} style={{ color: '#22c55e' }} />
        <h3 className="section-title mb-0">Model Comparison</h3>
        <div className="ml-auto flex items-center gap-1 badge badge-green">
          <Trophy size={11} />
          {MODEL_COLORS[bestModel]?.label || bestModel} best
        </div>
      </div>

      {/* ── Metric selector tabs ────────────────────────────────── */}
      <div className="flex gap-1 mb-4 p-1 rounded-xl" style={{ background: '#0a1a0f' }}>
        {Object.entries(METRIC_LABELS).map(([key, lbl]) => (
          <button
            key={key}
            id={`metric-tab-${key}`}
            onClick={() => setActiveMetric(key)}
            className="flex-1 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200"
            style={{
              background: activeMetric === key ? '#16a34a33' : 'transparent',
              color:      activeMetric === key ? '#22c55e' : '#64748b',
              border:     activeMetric === key ? '1px solid #16a34a66' : '1px solid transparent',
            }}
          >
            {lbl}
          </button>
        ))}
      </div>

      {/* ── Bar Chart ──────────────────────────────────────────── */}
      <div style={{ height: '180px' }}>
        <Bar data={chartData} options={chartOptions} />
      </div>

      {/* ── Metrics Table ──────────────────────────────────────── */}
      <div className="mt-5 overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: '1px solid #1a3826' }}>
              <th className="text-left pb-2 font-semibold" style={{ color: '#4a7c5e' }}>Model</th>
              {Object.values(METRIC_LABELS).map(lbl => (
                <th key={lbl} className="text-center pb-2 font-semibold" style={{ color: '#4a7c5e' }}>{lbl}</th>
              ))}
              {allModelsPrediction && (
                <th className="text-center pb-2 font-semibold" style={{ color: '#4a7c5e' }}>Predicted</th>
              )}
            </tr>
          </thead>
          <tbody>
            {modelNames.map((name) => {
              const m   = metrics[name];
              const col = MODEL_COLORS[name];
              const isRF = name === 'RandomForest';
              const predResult = allModelsPrediction?.[name];

              return (
                <tr
                  key={name}
                  style={{
                    borderBottom:  '1px solid #0a1a0f',
                    background:    isRF ? '#16a34a0a' : 'transparent',
                  }}
                >
                  <td className="py-2.5 pr-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ background: col?.bar }} />
                      <span style={{ color: '#e2e8f0' }}>{col?.label || name}</span>
                      {isRF && (
                        <span className="badge badge-green text-[10px] px-1.5 py-0.5">Primary</span>
                      )}
                    </div>
                  </td>
                  {Object.keys(METRIC_LABELS).map(key => (
                    <td key={key} className="text-center py-2.5" style={{ color: '#94a3b8' }}>
                      {m[key] != null ? `${(m[key] * 100).toFixed(2)}%` : '—'}
                    </td>
                  ))}
                  {allModelsPrediction && (
                    <td className="text-center py-2.5 capitalize" style={{ color: col?.bar }}>
                      {predResult?.crop || '—'}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* ── Model info note ─────────────────────────────────────── */}
      <div className="mt-4 flex items-center gap-2 text-xs" style={{ color: '#4a7c5e' }}>
        <Cpu size={11} />
        <span>Random Forest is the primary prediction model. Others shown for comparison.</span>
      </div>
    </div>
  );
}
