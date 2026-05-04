/**
 * src/components/LoadingSpinner.jsx
 * Animated loading states used during API calls.
 */

import React from 'react';
import { Sprout } from 'lucide-react';

/** Full-page overlay spinner */
export function PageLoader() {
  return (
    <div className="fixed inset-0 flex items-center justify-center z-50"
         style={{ background: 'rgba(4, 13, 8, 0.85)', backdropFilter: 'blur(8px)' }}>
      <div className="flex flex-col items-center gap-4">
        <div className="relative">
          {/* Outer ring */}
          <div className="w-16 h-16 rounded-full border-4 border-green-900 border-t-green-400 animate-spin" />
          {/* Icon center */}
          <div className="absolute inset-0 flex items-center justify-center">
            <Sprout size={22} className="text-green-400 animate-pulse" />
          </div>
        </div>
        <p className="text-sm font-medium" style={{ color: '#86efac' }}>
          Analysing conditions…
        </p>
        <p className="text-xs" style={{ color: '#4a7c5e' }}>
          Running ML model + SHAP explanation
        </p>
      </div>
    </div>
  );
}

/** Inline spinner (small, inline with text) */
export function InlineSpinner({ size = 16, color = '#22c55e' }) {
  return (
    <div
      className="animate-spin rounded-full border-2"
      style={{
        width: size, height: size,
        borderColor: `${color}40`,
        borderTopColor: color,
      }}
    />
  );
}

/** Skeleton card placeholder (shimmer) */
export function SkeletonCard({ className = '' }) {
  return (
    <div className={`card p-6 ${className}`}>
      <div className="shimmer h-4 w-2/3 rounded-lg mb-3" />
      <div className="shimmer h-8 w-1/2 rounded-lg mb-2" />
      <div className="shimmer h-3 w-full rounded-lg mb-2" />
      <div className="shimmer h-3 w-3/4 rounded-lg" />
    </div>
  );
}

export default PageLoader;
