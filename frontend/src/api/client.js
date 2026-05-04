/**
 * src/api/client.js
 * Axios client configured to point at the FastAPI backend.
 * 
 * In development: requests go to http://localhost:8000 via Vite proxy
 * In production: set VITE_API_URL environment variable
 */

import axios from 'axios';

// Base URL: use env variable or fall back to localhost
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: `${BASE_URL}/api/v1`,
  timeout: 30000,       // 30s timeout (SHAP can be slow on first call)
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── API Functions ──────────────────────────────────────────────────────────

/**
 * POST /predict
 * Send 7 soil/env features and get crop prediction from all 3 models.
 */
export const predictCrop = (inputData) =>
  apiClient.post('/predict', inputData).then(r => r.data);

/**
 * POST /explain
 * Get full SHAP explanation for the prediction.
 */
export const explainPrediction = (inputData) =>
  apiClient.post('/explain', inputData).then(r => r.data);

/**
 * GET /metrics
 * Get model evaluation metrics (accuracy, F1, etc.) for all 3 models.
 */
export const getMetrics = () =>
  apiClient.get('/metrics').then(r => r.data);

/**
 * GET /feature-importance
 * Get global RF feature importance (fast, no SHAP needed).
 */
export const getFeatureImportance = () =>
  apiClient.get('/feature-importance').then(r => r.data);

/**
 * GET /health
 * Check if the backend is alive and models are loaded.
 */
export const checkHealth = () =>
  apiClient.get('/health').then(r => r.data);

export default apiClient;
