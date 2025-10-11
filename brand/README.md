# MirrorCore-X Brand Assets

## 🎨 Brand Overview

MirrorCore-X is an advanced multi-agent cognitive trading system. The brand identity reflects its AI-powered capabilities, precision trading, and futuristic technology through a cyberpunk-inspired visual language with neon glows and neural network visualization.

## 📁 Asset Files

### Logo Variants (All SVG)
- `logo.svg` - Primary circular logo with neural network visualization
- `logo-horizontal.svg` - Horizontal logo variant with full brand name
- `logo-icon.svg` - Standalone "M" symbol for favicons and app icons
- `light/` - Light variants for dark backgrounds
- `dark/` - Dark variants for light backgrounds

### Social Media Assets
- `social/profile-200.svg` - Profile image (200×200, Twitter/GitHub)
- `social/profile-400.svg` - High-res profile image (400×400, LinkedIn)
- `social/twitter-banner.svg` - Twitter header (1500×500)
- `social/linkedin-banner.svg` - LinkedIn background (1584×396)
- `social/github-banner.svg` - GitHub profile banner (1280×640)

### Favicon Package
- `favicon/favicon-16x16.svg` - Browser tab icon (16×16)
- `favicon/favicon-32x32.svg` - Browser tab icon (32×32)
- `favicon/apple-touch-icon.svg` - iOS home screen (180×180)
- `favicon/android-chrome-192x192.svg` - Android icon (192×192)
- `favicon/android-chrome-512x512.svg` - Android icon (512×512)
- `favicon/site.webmanifest` - PWA manifest

### Documentation
- `style-guide.json` - Complete brand specifications in JSON
- `index.html` - Interactive brand showcase (http://0.0.0.0:5000/)

## 🎨 Color Palette

### Primary Colors
- **Cyan**: `#00D9FF` - Primary brand color, used for interactive elements
- **Purple**: `#7B2FFF` - Secondary brand color, used for accents
- **Dark**: `#0A0E27` - Primary background
- **Dark Secondary**: `#1A1F3A` - Secondary background

### Accent Colors
- **Success**: `#00FF88` - Positive states, profit indicators
- **Warning**: `#FFB800` - Alerts, caution states
- **Error**: `#FF3366` - Errors, loss indicators
- **Neutral**: `#8B92B0` - Muted text, disabled states

### Gradients
```css
/* Primary Gradient */
background: linear-gradient(135deg, #00D9FF 0%, #7B2FFF 100%);

/* Background Gradient */
background: linear-gradient(135deg, #0A0E27 0%, #1A1F3A 100%);

/* Accent Gradient */
background: linear-gradient(90deg, #00F0FF 0%, #9D4EFF 100%);
```

## 🔤 Typography

### Font Families
- **Display**: Orbitron (Brand name, major headings)
- **Headings**: Rajdhani (Section titles, subheadings)
- **Body**: Inter (Body text, UI elements)
- **Monospace**: JetBrains Mono (Code, technical data)

### Import Fonts
```html
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

## ✨ Visual Effects

### Glass Effect
```css
background: rgba(26, 31, 58, 0.7);
backdrop-filter: blur(10px);
border: 1px solid rgba(0, 217, 255, 0.2);
```

### Neon Glow
```css
text-shadow: 0 0 10px #00D9FF, 0 0 20px #00D9FF, 0 0 30px #7B2FFF;
box-shadow: 0 0 15px rgba(0, 217, 255, 0.5), 0 0 30px rgba(123, 47, 255, 0.3);
```

## 📐 Logo Usage Guidelines

1. **Clear Space**: Maintain minimum clear space equal to the height of the "X" around the logo
2. **Minimum Size**: 
   - Logo: 100px minimum width
   - Icon: 32px minimum size
3. **Backgrounds**: 
   - Preferred on dark backgrounds (#0A0E27, #1A1F3A)
   - Ensure sufficient contrast
4. **Alterations**: Do not modify colors, proportions, or add effects to the logo

## 🖼️ Logo Variants

### Primary Logo (`logo.svg`)
- Square format (200x200)
- Full mirror effect with neural network visualization
- Use for: Social media, app icons, standalone branding

### Horizontal Logo (`logo-horizontal.svg`)
- Wide format (400x120)
- Includes full brand name and tagline
- Use for: Website headers, presentations, banners

### Icon (`icon.svg`)
- Simplified square format (64x64)
- Core mirror symbol with X mark
- Use for: Favicons, app icons, small UI elements

## 🎯 Brand Personality

- **Sophisticated**: Advanced AI technology
- **Precise**: Data-driven decision making
- **Futuristic**: Cutting-edge trading system
- **Trustworthy**: Reliable cognitive analysis
- **Powerful**: Multi-agent architecture

## 🚀 Implementation Status

### ✅ Completed
- **Brand Identity System**: Full logo suite with variants and social assets
- **Design System**: Unified theme with Tailwind integration
- **React Frontend**: All pages themed with brand consistency
  - Dashboard with real-time metrics
  - Trading signals interface
  - Performance analytics
  - Risk management UI
  - Strategy management
  - Settings and configuration
- **Web Assets**: Complete favicon package and meta tags
- **Brand Showcase**: Interactive demo at http://0.0.0.0:5000/

### Component Utilities
Custom CSS utilities available throughout the app:
- `.brand-card` - Glass morphism card
- `.metric-card` - Metrics display with glow
- `.chart-container` - Chart wrapper
- `.page-header` - Page title section
- `.btn-primary` - Primary action button
- `.status-active` / `.status-inactive` - Status indicators
- `.signal-buy` / `.signal-sell` - Trading signal cards

## 🚀 Quick Start

### View Brand Assets
The brand showcase server is running at **http://0.0.0.0:5000/** with interactive examples of all assets.

### Use in Your Project
1. Copy SVG files from `brand/` to your assets directory
2. Import fonts in your HTML (see Typography section)
3. Reference colors from `style-guide.json` or Tailwind config
4. Use utility classes from `src/index.css`
5. Follow usage guidelines for brand consistency

### Integration Points
- **Tailwind Config**: Brand tokens in `tailwind.config.js`
- **CSS Utilities**: Component classes in `src/index.css`
- **React Theme**: TypeScript types in `src/theme/brand.ts`
- **HTML Meta**: Favicons and OG tags in `index.html`

## 📄 License

These brand assets are part of the MirrorCore-X project. Please maintain brand consistency when using these assets.

---

**MirrorCore-X** - Where AI Meets Trading Intelligence
