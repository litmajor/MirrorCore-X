export const brandColors = {
  primary: {
    cyan: '#00D9FF',
    purple: '#7B2FFF',
    dark: '#0A0E27',
    darkSecondary: '#1A1F3A',
  },
  secondary: {
    electricBlue: '#00F0FF',
    neonPurple: '#9D4EFF',
    midnightBlue: '#141B3D',
  },
  accents: {
    success: '#00FF88',
    warning: '#FFB800',
    error: '#FF3366',
    neutral: '#8B92B0',
  },
  backgrounds: {
    primary: '#0A0E27',
    secondary: '#141B3D',
    tertiary: '#1F2847',
    surface: '#2A3150',
  },
  text: {
    primary: '#FFFFFF',
    secondary: '#B8BFDC',
    muted: '#6B7299',
    accent: '#00D9FF',
  },
};

export const brandGradients = {
  primary: 'linear-gradient(135deg, #00D9FF 0%, #7B2FFF 100%)',
  secondary: 'linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%)',
  accent: 'linear-gradient(90deg, #00F0FF 0%, #9D4EFF 100%)',
  overlay: 'linear-gradient(180deg, rgba(10, 14, 39, 0) 0%, rgba(10, 14, 39, 0.8) 100%)',
};

export const brandFonts = {
  display: "'Orbitron', 'Rajdhani', 'Exo 2', sans-serif",
  heading: "'Rajdhani', 'Inter', sans-serif",
  body: "'Inter', 'Roboto', system-ui, sans-serif",
  mono: "'JetBrains Mono', 'Fira Code', monospace",
};

export const brandEffects = {
  glass: {
    background: 'rgba(26, 31, 58, 0.7)',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(0, 217, 255, 0.2)',
  },
  neon: {
    textShadow: '0 0 10px #00D9FF, 0 0 20px #00D9FF, 0 0 30px #7B2FFF',
    boxShadow: '0 0 15px rgba(0, 217, 255, 0.5), 0 0 30px rgba(123, 47, 255, 0.3)',
  },
  glow: {
    boxShadow: '0 0 20px rgba(0, 217, 255, 0.4), 0 0 40px rgba(123, 47, 255, 0.3)',
  },
};

export const brandShadows = {
  sm: '0 2px 8px rgba(0, 217, 255, 0.1)',
  md: '0 4px 16px rgba(0, 217, 255, 0.15)',
  lg: '0 8px 32px rgba(0, 217, 255, 0.2)',
  glow: '0 0 20px rgba(0, 217, 255, 0.4), 0 0 40px rgba(123, 47, 255, 0.3)',
};
