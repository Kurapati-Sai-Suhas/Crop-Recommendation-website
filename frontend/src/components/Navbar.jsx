/**
 * src/components/Navbar.jsx
 * Top navigation bar with branding and page links.
 */

import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Sprout, BarChart3, Home, Menu, X } from 'lucide-react';

export default function Navbar() {
  const location  = useLocation();
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  // Add shadow when user scrolls down
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const navLinks = [
    { to: '/',          label: 'Home',       icon: Home },
    { to: '/dashboard', label: 'Dashboard',  icon: BarChart3 },
  ];

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
      style={{
        background: scrolled
          ? 'rgba(4, 13, 8, 0.95)'
          : 'rgba(4, 13, 8, 0.7)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(26, 56, 38, 0.6)',
        boxShadow: scrolled ? '0 4px 30px rgba(0,0,0,0.4)' : 'none',
      }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16">

          {/* ── Brand Logo ── */}
          <Link to="/" className="flex items-center gap-2.5 group" id="nav-logo">
            <div className="p-2 rounded-xl" style={{ background: '#16a34a22', border: '1px solid #16a34a44' }}>
              <Sprout size={20} className="text-green-400 group-hover:text-green-300 transition-colors" />
            </div>
            <div>
              <span className="font-bold text-base" style={{ color: '#86efac' }}>CropAI</span>
              <span className="text-xs block leading-none" style={{ color: '#4a7c5e' }}>Explainable AI</span>
            </div>
          </Link>

          {/* ── Desktop Nav Links ── */}
          <div className="hidden md:flex items-center gap-1">
            {navLinks.map(({ to, label, icon: Icon }) => {
              const isActive = location.pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  id={`nav-${label.toLowerCase()}`}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200"
                  style={{
                    color:      isActive ? '#22c55e' : '#94a3b8',
                    background: isActive ? '#16a34a22' : 'transparent',
                    border:     isActive ? '1px solid #16a34a44' : '1px solid transparent',
                  }}
                >
                  <Icon size={15} />
                  {label}
                </Link>
              );
            })}

            {/* CTA Button */}
            <Link
              to="/dashboard"
              id="nav-cta"
              className="btn-primary ml-3 text-xs px-4 py-2"
            >
              Try Now →
            </Link>
          </div>

          {/* ── Mobile Menu Toggle ── */}
          <button
            className="md:hidden p-2 rounded-lg"
            style={{ color: '#94a3b8', border: '1px solid #1a3826' }}
            onClick={() => setMenuOpen(!menuOpen)}
            id="nav-mobile-toggle"
          >
            {menuOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>

        {/* ── Mobile Dropdown ── */}
        {menuOpen && (
          <div className="md:hidden pb-4 pt-2 flex flex-col gap-1 animate-fade-in">
            {navLinks.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm"
                style={{ color: '#94a3b8', border: '1px solid #1a3826' }}
                onClick={() => setMenuOpen(false)}
              >
                <Icon size={16} />
                {label}
              </Link>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
}
