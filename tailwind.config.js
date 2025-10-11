/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary Brand Colors
        brand: {
          cyan: '#00D9FF',
          purple: '#7B2FFF',
          dark: '#0A0E27',
          'dark-secondary': '#1A1F3A',
        },
        // Secondary Colors
        electric: {
          blue: '#00F0FF',
          purple: '#9D4EFF',
        },
        midnight: {
          blue: '#141B3D',
        },
        // Accent Colors
        success: '#00FF88',
        warning: '#FFB800',
        error: '#FF3366',
        neutral: '#8B92B0',
        // Background Colors
        bg: {
          primary: '#0A0E27',
          secondary: '#141B3D',
          tertiary: '#1F2847',
          surface: '#2A3150',
        },
        // Text Colors
        txt: {
          primary: '#FFFFFF',
          secondary: '#B8BFDC',
          muted: '#6B7299',
          accent: '#00D9FF',
        },
      },
      fontFamily: {
        display: ['Orbitron', 'Rajdhani', 'Exo 2', 'sans-serif'],
        heading: ['Rajdhani', 'Inter', 'sans-serif'],
        body: ['Inter', 'Roboto', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #00D9FF 0%, #7B2FFF 100%)',
        'gradient-secondary': 'linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%)',
        'gradient-accent': 'linear-gradient(90deg, #00F0FF 0%, #9D4EFF 100%)',
        'gradient-overlay': 'linear-gradient(180deg, rgba(10, 14, 39, 0) 0%, rgba(10, 14, 39, 0.8) 100%)',
      },
      boxShadow: {
        'brand-sm': '0 2px 8px rgba(0, 217, 255, 0.1)',
        'brand-md': '0 4px 16px rgba(0, 217, 255, 0.15)',
        'brand-lg': '0 8px 32px rgba(0, 217, 255, 0.2)',
        'brand-glow': '0 0 20px rgba(0, 217, 255, 0.4), 0 0 40px rgba(123, 47, 255, 0.3)',
        'neon': '0 0 15px rgba(0, 217, 255, 0.5), 0 0 30px rgba(123, 47, 255, 0.3)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(0, 217, 255, 0.4), 0 0 40px rgba(123, 47, 255, 0.3)' },
          '100%': { boxShadow: '0 0 30px rgba(0, 217, 255, 0.6), 0 0 60px rgba(123, 47, 255, 0.5)' },
        },
      },
    },
  },
  plugins: [],
}
