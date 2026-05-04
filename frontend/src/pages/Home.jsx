/**
 * src/pages/Home.jsx
 * Landing page with hero section, feature highlights, and CTA.
 */

import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Sprout, Brain, BarChart2, ShieldCheck,
  Cpu, FlaskConical, Zap, ArrowRight, CheckCircle
} from 'lucide-react';
import { checkHealth } from '../api/client';

// ── Feature highlight cards data ───────────────────────────────────────────
const FEATURES = [
  {
    icon: Brain,
    color: '#22c55e',
    bg: '#22c55e15',
    title: 'Explainable AI (SHAP)',
    desc: 'Every prediction comes with SHAP values showing exactly which soil conditions drove the recommendation.',
  },
  {
    icon: BarChart2,
    color: '#3b82f6',
    bg: '#3b82f615',
    title: 'Model Comparison',
    desc: 'Compare Random Forest, Logistic Regression, and Naive Bayes performance side-by-side in real time.',
  },
  {
    icon: Cpu,
    color: '#a855f7',
    bg: '#a855f715',
    title: 'MLOps Pipeline',
    desc: 'Full MLflow experiment tracking, DVC data versioning, and GitHub Actions CI/CD baked in.',
  },
  {
    icon: ShieldCheck,
    color: '#f59e0b',
    bg: '#f59e0b15',
    title: 'Input Validation',
    desc: 'Pydantic-powered backend validation with descriptive error messages. Safe and production-ready.',
  },
  {
    icon: FlaskConical,
    color: '#ec4899',
    bg: '#ec489915',
    title: '22 Crop Classes',
    desc: 'Trained on rice, maize, banana, coffee, cotton, and 17 more crops with >95% accuracy.',
  },
  {
    icon: Zap,
    color: '#0ea5e9',
    bg: '#0ea5e915',
    title: 'Fast Inference',
    desc: 'Sub-second predictions with pre-loaded models. SHAP explanations in under 2 seconds.',
  },
];

// ── Crops showcase (22 crops with emoji) ──────────────────────────────────
const CROPS = [
  { name: 'Rice',        emoji: '🌾' },
  { name: 'Maize',       emoji: '🌽' },
  { name: 'Banana',      emoji: '🍌' },
  { name: 'Mango',       emoji: '🥭' },
  { name: 'Coffee',      emoji: '☕' },
  { name: 'Cotton',      emoji: '🌸' },
  { name: 'Grapes',      emoji: '🍇' },
  { name: 'Coconut',     emoji: '🥥' },
  { name: 'Watermelon',  emoji: '🍉' },
  { name: 'Apple',       emoji: '🍎' },
  { name: 'Orange',      emoji: '🍊' },
  { name: 'Jute',        emoji: '🌿' },
];

// ── Stats ─────────────────────────────────────────────────────────────────
const STATS = [
  { value: '22',   label: 'Crop Classes' },
  { value: '2200', label: 'Training Samples' },
  { value: '95%+', label: 'RF Accuracy' },
  { value: '7',    label: 'Input Features' },
];

export default function Home() {
  const [backendStatus, setBackendStatus] = useState(null); // null | 'ok' | 'error'

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then(() => setBackendStatus('ok'))
      .catch(() => setBackendStatus('error'));
  }, []);

  return (
    <div className="min-h-screen" style={{ background: 'var(--color-bg)' }}>

      {/* ══ HERO SECTION ═══════════════════════════════════════════════════ */}
      <section
        className="relative pt-32 pb-24 px-4 overflow-hidden"
        style={{ backgroundImage: 'var(--tw-gradient-stops)' }}
      >
        {/* Glowing background blobs */}
        <div
          className="absolute top-20 left-1/2 -translate-x-1/2 w-[600px] h-[600px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(circle, #16a34a18 0%, transparent 70%)',
            filter: 'blur(40px)',
          }}
        />
        <div
          className="absolute top-40 right-10 w-[300px] h-[300px] rounded-full pointer-events-none"
          style={{
            background: 'radial-gradient(circle, #22c55e0a 0%, transparent 70%)',
            filter: 'blur(60px)',
          }}
        />

        <div className="max-w-5xl mx-auto text-center relative z-10">

          {/* Status badge */}
          <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full text-xs font-semibold"
               style={{ background: '#16a34a15', border: '1px solid #16a34a33', color: '#4ade80' }}>
            <span className="w-2 h-2 rounded-full animate-pulse"
                  style={{ background: backendStatus === 'ok' ? '#22c55e' : backendStatus === 'error' ? '#ef4444' : '#f59e0b' }} />
            {backendStatus === 'ok'
              ? 'Backend online — Models loaded'
              : backendStatus === 'error'
              ? 'Backend offline — Start the server'
              : 'Connecting to backend…'}
          </div>

          {/* Main heading */}
          <h1 className="text-5xl sm:text-6xl font-extrabold leading-tight mb-6">
            <span style={{ color: '#e2e8f0' }}>Crop Recommendation</span>
            <br />
            <span className="gradient-text">with Explainable AI</span>
          </h1>

          <p className="text-lg max-w-2xl mx-auto mb-10 leading-relaxed" style={{ color: '#94a3b8' }}>
            Enter your soil and climate conditions and get an AI-powered crop recommendation —
            with full <strong style={{ color: '#86efac' }}>SHAP explanations</strong> telling you
            exactly <em>why</em> that crop was chosen.
          </p>

          {/* CTA buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <Link
              to="/dashboard"
              id="hero-cta-primary"
              className="btn-primary text-base px-8 py-4 rounded-2xl"
            >
              <Sprout size={20} />
              Get Crop Recommendation
              <ArrowRight size={16} />
            </Link>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              id="hero-cta-docs"
              className="btn-secondary text-base px-8 py-4 rounded-2xl"
            >
              View API Docs →
            </a>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-2xl mx-auto">
            {STATS.map(({ value, label }) => (
              <div
                key={label}
                className="rounded-2xl px-4 py-4 text-center"
                style={{ background: '#0d1f13', border: '1px solid #1a3826' }}
              >
                <div className="text-2xl font-extrabold gradient-text">{value}</div>
                <div className="text-xs mt-1" style={{ color: '#4a7c5e' }}>{label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ CROP SHOWCASE ══════════════════════════════════════════════════ */}
      <section className="py-12 px-4 overflow-hidden">
        <div className="max-w-5xl mx-auto">
          <p className="text-center text-xs font-semibold uppercase tracking-widest mb-6"
             style={{ color: '#4a7c5e' }}>
            Supported Crops
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            {CROPS.map(({ name, emoji }) => (
              <div
                key={name}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 hover:scale-105 cursor-default"
                style={{ background: '#0d1f13', border: '1px solid #1a3826', color: '#94a3b8' }}
              >
                <span>{emoji}</span>
                {name}
              </div>
            ))}
            <div className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm"
                 style={{ background: '#16a34a15', border: '1px solid #16a34a33', color: '#4ade80' }}>
              + 10 more
            </div>
          </div>
        </div>
      </section>

      {/* ══ FEATURES GRID ══════════════════════════════════════════════════ */}
      <section className="py-16 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-3" style={{ color: '#e2e8f0' }}>
              Production-Ready MLOps Stack
            </h2>
            <p style={{ color: '#64748b' }}>
              Built for academic presentation and real-world deployment
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {FEATURES.map(({ icon: Icon, color, bg, title, desc }) => (
              <div
                key={title}
                className="card p-6 transition-all duration-300 hover:scale-[1.02]"
                style={{ cursor: 'default' }}
              >
                <div className="inline-flex p-3 rounded-xl mb-4" style={{ background: bg }}>
                  <Icon size={20} style={{ color }} />
                </div>
                <h3 className="font-semibold text-sm mb-2" style={{ color: '#e2e8f0' }}>{title}</h3>
                <p className="text-xs leading-relaxed" style={{ color: '#64748b' }}>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ══ HOW IT WORKS ═══════════════════════════════════════════════════ */}
      <section className="py-16 px-4">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-2xl font-bold text-center mb-10" style={{ color: '#e2e8f0' }}>
            How It Works
          </h2>

          {[
            { step: '01', title: 'Enter Conditions', desc: 'Input your soil NPK values, temperature, humidity, pH, and rainfall.' },
            { step: '02', title: 'ML Prediction',    desc: 'Random Forest model (100 trees) predicts the optimal crop with confidence score.' },
            { step: '03', title: 'SHAP Explanation', desc: 'SHAP TreeExplainer computes feature-level impact values for the prediction.' },
            { step: '04', title: 'Actionable Insight', desc: 'Get human-readable explanation + visual charts to understand the recommendation.' },
          ].map(({ step, title, desc }, i) => (
            <div key={step} className="flex gap-5 mb-6">
              <div className="shrink-0 w-12 h-12 rounded-2xl flex items-center justify-center font-extrabold text-sm"
                   style={{ background: '#16a34a22', border: '1px solid #16a34a44', color: '#22c55e' }}>
                {step}
              </div>
              <div className="pt-1">
                <h3 className="font-semibold text-sm mb-1" style={{ color: '#86efac' }}>{title}</h3>
                <p className="text-xs leading-relaxed" style={{ color: '#64748b' }}>{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ══ BOTTOM CTA ═════════════════════════════════════════════════════ */}
      <section className="py-16 px-4">
        <div className="max-w-2xl mx-auto text-center">
          <div
            className="rounded-3xl p-10"
            style={{
              background: 'linear-gradient(135deg, #0d1f13, #0a2b17)',
              border: '1px solid #1a3826',
              boxShadow: '0 0 60px #22c55e10',
            }}
          >
            <Sprout size={36} className="mx-auto mb-4" style={{ color: '#22c55e' }} />
            <h2 className="text-2xl font-bold mb-3" style={{ color: '#e2e8f0' }}>
              Ready to recommend your crop?
            </h2>
            <p className="text-sm mb-6" style={{ color: '#64748b' }}>
              Takes under 30 seconds. No sign-up required.
            </p>
            <Link
              to="/dashboard"
              id="bottom-cta"
              className="btn-primary inline-flex text-base px-8 py-4 rounded-2xl"
            >
              <Sprout size={18} />
              Open Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* ══ FOOTER ═════════════════════════════════════════════════════════ */}
      <footer className="py-8 px-4 text-center" style={{ borderTop: '1px solid #1a3826' }}>
        <p className="text-xs" style={{ color: '#2d5a3d' }}>
          CropAI · Explainable AI Crop Recommendation System ·{' '}
          Built with FastAPI · React · SHAP · MLflow · Docker
        </p>
      </footer>
    </div>
  );
}
