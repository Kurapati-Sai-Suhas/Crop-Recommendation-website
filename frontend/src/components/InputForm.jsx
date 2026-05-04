/**
 * src/components/InputForm.jsx
 * The 7-parameter soil input form with validation and tooltips.
 *
 * Features:
 *  - Real-time client-side validation (range checks)
 *  - Range sliders with numeric input fallback
 *  - Tooltip hints explaining each parameter
 *  - Demo "fill sample values" button
 */

import React, { useState } from 'react';
import { Sprout, Thermometer, Droplets, FlaskConical, CloudRain, Zap, Info } from 'lucide-react';
import { InlineSpinner } from './LoadingSpinner';

// ── Field metadata ──────────────────────────────────────────────────────────
const FIELDS = [
  {
    key: 'N', label: 'Nitrogen (N)', unit: 'ratio',
    min: 0, max: 200, step: 1, icon: Zap,
    color: '#22c55e',
    hint: 'Nitrogen content in the soil. Higher values indicate nitrogen-rich soil.',
    example: 80,
  },
  {
    key: 'P', label: 'Phosphorus (P)', unit: 'ratio',
    min: 0, max: 200, step: 1, icon: Zap,
    color: '#3b82f6',
    hint: 'Phosphorus content. Essential for root development and fruiting.',
    example: 40,
  },
  {
    key: 'K', label: 'Potassium (K)', unit: 'ratio',
    min: 0, max: 210, step: 1, icon: Zap,
    color: '#8b5cf6',
    hint: 'Potassium content. Improves disease resistance and water regulation.',
    example: 40,
  },
  {
    key: 'temperature', label: 'Temperature', unit: '°C',
    min: 0, max: 50, step: 0.1, icon: Thermometer,
    color: '#f59e0b',
    hint: 'Average ambient temperature in Celsius.',
    example: 23.0,
  },
  {
    key: 'humidity', label: 'Humidity', unit: '%',
    min: 0, max: 100, step: 0.1, icon: Droplets,
    color: '#06b6d4',
    hint: 'Relative humidity percentage in the air.',
    example: 82.0,
  },
  {
    key: 'ph', label: 'pH Level', unit: '',
    min: 0, max: 14, step: 0.1, icon: FlaskConical,
    color: '#ec4899',
    hint: 'Soil pH (0 = acidic, 7 = neutral, 14 = alkaline). Most crops prefer 6–7.',
    example: 6.0,
  },
  {
    key: 'rainfall', label: 'Rainfall', unit: 'mm',
    min: 0, max: 500, step: 1, icon: CloudRain,
    color: '#0ea5e9',
    hint: 'Annual rainfall in millimeters.',
    example: 200.0,
  },
];

// Sample values for quick demo
const SAMPLE_VALUES = {
  N: 80, P: 40, K: 40, temperature: 23.0, humidity: 82.0, ph: 6.0, rainfall: 200.0
};

export default function InputForm({ onSubmit, loading }) {
  // Form state
  const [values, setValues] = useState({
    N: '', P: '', K: '', temperature: '', humidity: '', ph: '', rainfall: ''
  });
  const [errors, setErrors] = useState({});
  const [tooltip, setTooltip] = useState(null);

  // ── Validation ─────────────────────────────────────────────────────────────
  const validate = (key, val) => {
    const field = FIELDS.find(f => f.key === key);
    if (!field) return '';
    const num = parseFloat(val);
    if (val === '' || isNaN(num)) return 'This field is required';
    if (num < field.min) return `Minimum value is ${field.min}`;
    if (num > field.max) return `Maximum value is ${field.max}`;
    return '';
  };

  const handleChange = (key, val) => {
    setValues(prev => ({ ...prev, [key]: val }));
    const err = validate(key, val);
    setErrors(prev => ({ ...prev, [key]: err }));
  };

  // Fill with sample values
  const fillSample = () => {
    const strValues = {};
    Object.entries(SAMPLE_VALUES).forEach(([k, v]) => { strValues[k] = String(v); });
    setValues(strValues);
    setErrors({});
  };

  // ── Submit ─────────────────────────────────────────────────────────────────
  const handleSubmit = (e) => {
    e.preventDefault();

    // Validate all fields
    const newErrors = {};
    FIELDS.forEach(({ key }) => {
      newErrors[key] = validate(key, values[key]);
    });
    setErrors(newErrors);

    if (Object.values(newErrors).some(Boolean)) return;

    // Convert to floats and call parent
    const payload = {};
    FIELDS.forEach(({ key }) => { payload[key] = parseFloat(values[key]); });
    onSubmit(payload);
  };

  const allFilled = FIELDS.every(({ key }) => values[key] !== '');

  return (
    <form onSubmit={handleSubmit} id="crop-input-form" noValidate>
      <div className="card p-6">

        {/* ── Header ─────────────────────────────────────────────── */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="font-semibold text-base" style={{ color: '#86efac' }}>
              Soil & Environment Parameters
            </h2>
            <p className="text-xs mt-0.5" style={{ color: '#4a7c5e' }}>
              Enter all 7 conditions for accurate recommendation
            </p>
          </div>
          <button
            type="button"
            onClick={fillSample}
            id="fill-sample-btn"
            className="btn-secondary text-xs px-3 py-2"
          >
            <Sprout size={13} />
            Load Sample
          </button>
        </div>

        {/* ── Fields Grid ────────────────────────────────────────── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {FIELDS.map(({ key, label, unit, min, max, step, icon: Icon, color, hint }) => (
            <div key={key} className="relative">
              <label
                htmlFor={`input-${key}`}
                className="flex items-center gap-1.5 text-xs font-semibold mb-1.5"
                style={{ color: '#94a3b8' }}
              >
                <Icon size={12} style={{ color }} />
                {label}
                {unit && <span className="opacity-60">({unit})</span>}
                {/* Tooltip icon */}
                <button
                  type="button"
                  className="ml-auto opacity-50 hover:opacity-100 transition-opacity"
                  onMouseEnter={() => setTooltip(key)}
                  onMouseLeave={() => setTooltip(null)}
                >
                  <Info size={11} />
                </button>
              </label>

              {/* Tooltip */}
              {tooltip === key && (
                <div
                  className="absolute z-10 left-0 -top-2 transform -translate-y-full text-xs px-3 py-2 rounded-lg pointer-events-none animate-fade-in"
                  style={{
                    background: '#1a3020',
                    border: '1px solid #16a34a44',
                    color: '#86efac',
                    width: '220px',
                    boxShadow: '0 8px 20px rgba(0,0,0,0.5)',
                  }}
                >
                  {hint}
                </div>
              )}

              {/* Input */}
              <input
                id={`input-${key}`}
                type="number"
                min={min}
                max={max}
                step={step}
                value={values[key]}
                onChange={e => handleChange(key, e.target.value)}
                placeholder={`${min} – ${max}`}
                className="input-field"
                style={{
                  borderColor: errors[key]
                    ? '#ef4444'
                    : values[key] !== '' && !errors[key]
                    ? `${color}66`
                    : undefined,
                }}
              />

              {/* Range hint bar */}
              {values[key] !== '' && !errors[key] && (
                <div className="mt-1.5 h-1 rounded-full overflow-hidden" style={{ background: '#1a3020' }}>
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${Math.min(100, ((parseFloat(values[key]) - min) / (max - min)) * 100)}%`,
                      background: `linear-gradient(90deg, ${color}88, ${color})`,
                    }}
                  />
                </div>
              )}

              {/* Error message */}
              {errors[key] && (
                <p className="error-text" id={`error-${key}`}>{errors[key]}</p>
              )}
            </div>
          ))}
        </div>

        {/* ── Submit Button ───────────────────────────────────────── */}
        <button
          type="submit"
          id="predict-btn"
          disabled={loading || !allFilled}
          className="btn-primary w-full mt-6 py-4 text-base"
        >
          {loading ? (
            <>
              <InlineSpinner size={18} color="#fff" />
              Analysing…
            </>
          ) : (
            <>
              <Sprout size={18} />
              Recommend Crop
            </>
          )}
        </button>
      </div>
    </form>
  );
}
