/**
 * src/pages/Dashboard.jsx
 * Main prediction dashboard — the heart of the application.
 *
 * Layout (two-column on desktop):
 *   LEFT:  InputForm
 *   RIGHT: ResultCard → ModelComparison → ExplanationPanel
 *
 * Flow:
 *   1. User fills in 7 parameters and clicks "Recommend Crop"
 *   2. POST /predict → show ResultCard + ModelComparison
 *   3. POST /explain → show ExplanationPanel with SHAP values
 */

import React, { useState } from 'react';
import { AlertCircle, RotateCcw, Activity } from 'lucide-react';
import InputForm      from '../components/InputForm';
import ResultCard     from '../components/ResultCard';
import ModelComparison from '../components/ModelComparison';
import ExplanationPanel from '../components/ExplanationPanel';
import { PageLoader }  from '../components/LoadingSpinner';
import { predictCrop, explainPrediction } from '../api/client';

export default function Dashboard() {
  // ── State ───────────────────────────────────────────────────────────────
  const [loading,      setLoading]      = useState(false);
  const [explLoading,  setExplLoading]  = useState(false);
  const [prediction,   setPrediction]   = useState(null);   // /predict response
  const [explanation,  setExplanation]  = useState(null);   // /explain response
  const [error,        setError]        = useState(null);

  // ── Handle form submit ──────────────────────────────────────────────────
  const handleSubmit = async (formData) => {
    setError(null);
    setPrediction(null);
    setExplanation(null);

    // ── Step 1: Get prediction from all 3 models ─────────────────────────
    setLoading(true);
    try {
      const predResult = await predictCrop(formData);
      setPrediction(predResult);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Prediction failed';
      setError(`Prediction error: ${msg}`);
      setLoading(false);
      return;
    }
    setLoading(false);

    // ── Step 2: Get SHAP explanation (separate call — can take ~2s) ──────
    setExplLoading(true);
    try {
      const explResult = await explainPrediction(formData);
      setExplanation(explResult);
    } catch (err) {
      // Explanation failure is non-fatal — show prediction without explanation
      console.warn('Explanation failed:', err);
    }
    setExplLoading(false);
  };

  // ── Reset everything ────────────────────────────────────────────────────
  const handleReset = () => {
    setPrediction(null);
    setExplanation(null);
    setError(null);
  };

  const hasResults = prediction !== null;

  return (
    <div className="min-h-screen pt-20 pb-12 px-4" style={{ background: 'var(--color-bg)' }}>
      {/* ── Loading overlay ─────────────────────────────────────────── */}
      {loading && <PageLoader />}

      <div className="max-w-7xl mx-auto">

        {/* ── Page Header ─────────────────────────────────────────────── */}
        <div className="mb-8 flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold" style={{ color: '#86efac' }}>
              Prediction Dashboard
            </h1>
            <p className="text-sm mt-1" style={{ color: '#4a7c5e' }}>
              Fill in your soil & climate conditions to get a crop recommendation with SHAP explanations
            </p>
          </div>

          {hasResults && (
            <button
              onClick={handleReset}
              id="reset-btn"
              className="btn-secondary flex items-center gap-2 text-xs"
            >
              <RotateCcw size={13} />
              New Prediction
            </button>
          )}
        </div>

        {/* ── Error Banner ─────────────────────────────────────────────── */}
        {error && (
          <div
            className="mb-6 flex items-center gap-3 px-5 py-4 rounded-2xl animate-fade-in"
            style={{ background: '#ef444415', border: '1px solid #ef444433' }}
            id="error-banner"
          >
            <AlertCircle size={16} style={{ color: '#ef4444' }} />
            <p className="text-sm" style={{ color: '#fca5a5' }}>{error}</p>
            <button
              className="ml-auto text-xs underline"
              style={{ color: '#ef4444' }}
              onClick={() => setError(null)}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* ── Main Two-Column Layout ────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* ── LEFT COLUMN: Input Form ─────────────────────────────────── */}
          <div className="space-y-4">
            <InputForm onSubmit={handleSubmit} loading={loading} />

            {/* Quick tips card */}
            {!hasResults && (
              <div
                className="card p-5 animate-fade-in"
                style={{ borderColor: '#1a3826' }}
              >
                <p className="text-xs font-semibold mb-3 uppercase tracking-widest"
                   style={{ color: '#4a7c5e' }}>
                  💡 Parameter Ranges Guide
                </p>
                <div className="grid grid-cols-2 gap-2 text-xs" style={{ color: '#64748b' }}>
                  {[
                    ['Nitrogen (N)',    '0 – 200'],
                    ['Phosphorus (P)',  '0 – 200'],
                    ['Potassium (K)',   '0 – 210'],
                    ['Temperature',    '0 – 50 °C'],
                    ['Humidity',       '0 – 100 %'],
                    ['pH',             '0 – 14'],
                    ['Rainfall',       '0 – 500 mm'],
                  ].map(([label, range]) => (
                    <div key={label} className="flex justify-between">
                      <span style={{ color: '#94a3b8' }}>{label}</span>
                      <span className="font-mono">{range}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* ── RIGHT COLUMN: Results ──────────────────────────────────── */}
          <div className="space-y-4" id="results-column">

            {/* Empty state */}
            {!hasResults && !loading && (
              <div
                className="card flex flex-col items-center justify-center py-20 text-center"
                id="empty-state"
              >
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                  style={{ background: '#16a34a10', border: '1px solid #1a3826' }}
                >
                  <Activity size={28} style={{ color: '#16a34a' }} />
                </div>
                <p className="font-semibold text-sm mb-1" style={{ color: '#4a7c5e' }}>
                  No prediction yet
                </p>
                <p className="text-xs" style={{ color: '#2d5a3d' }}>
                  Fill in the form and click "Recommend Crop"
                </p>
              </div>
            )}

            {/* ── Prediction result card ─────────────────────────────── */}
            {hasResults && (
              <ResultCard result={prediction} />
            )}

            {/* ── SHAP Explanation loading state ─────────────────────── */}
            {hasResults && explLoading && (
              <div
                className="card px-5 py-4 flex items-center gap-3 animate-fade-in"
                id="explanation-loading"
              >
                <div className="w-4 h-4 rounded-full border-2 border-green-900 border-t-green-400 animate-spin" />
                <p className="text-xs" style={{ color: '#4a7c5e' }}>
                  Generating SHAP explanation…
                </p>
              </div>
            )}
          </div>
        </div>

        {/* ── BOTTOM FULL-WIDTH: Model Comparison + Explanation ────────── */}
        {hasResults && (
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Model comparison */}
            <ModelComparison
              allModelsPrediction={prediction?.all_models}
            />

            {/* SHAP explanation */}
            {explanation && (
              <ExplanationPanel explanation={explanation} />
            )}

            {/* Placeholder while explanation loads */}
            {!explanation && !explLoading && hasResults && (
              <div className="card p-6 flex items-center justify-center text-center">
                <p className="text-xs" style={{ color: '#2d5a3d' }}>
                  Explanation panel will appear here after the SHAP computation completes.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
