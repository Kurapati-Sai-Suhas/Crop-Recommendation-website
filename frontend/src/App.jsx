/**
 * src/App.jsx
 * Root application component with React Router setup.
 * 
 * Routes:
 *   /           → Home page
 *   /dashboard  → Prediction Dashboard
 */

import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';

function App() {
  return (
    <BrowserRouter>
      {/* ── Persistent navigation bar ── */}
      <Navbar />

      {/* ── Page routes ── */}
      <Routes>
        <Route path="/"          element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        {/* Redirect unknown paths to home */}
        <Route path="*"          element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
