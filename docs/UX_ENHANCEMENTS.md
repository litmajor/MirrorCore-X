# UX Enhancements Guide

## Overview

This document outlines the user experience enhancements added to MirrorCore-X, including loading states, notifications, theming, and mobile responsiveness.

## 1. Loading Skeletons

### Purpose
Provide visual feedback while data is loading, improving perceived performance and user experience.

### Components

**Available Skeleton Components:**
- `Skeleton` - Base skeleton component with customizable variants
- `MetricCardSkeleton` - Loading state for metric cards
- `ChartSkeleton` - Loading state for charts
- `TableRowSkeleton` - Loading state for table rows
- `CardSkeleton` - Generic card loading state

### Usage Example

```tsx
import { MetricCardSkeleton, ChartSkeleton } from '../components/Skeleton';

function Dashboard() {
  const { data, loading } = useAPI('/api/data');

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
      {loading ? (
        <>
          <MetricCardSkeleton />
          <MetricCardSkeleton />
          <MetricCardSkeleton />
          <MetricCardSkeleton />
        </>
      ) : (
        // Render actual metric cards
        data.map(item => <MetricCard key={item.id} {...item} />)
      )}
    </div>
  );
}
```

### Customization

The base `Skeleton` component supports:
- **Variants**: `text`, `circular`, `rectangular`
- **Animation**: `pulse`, `wave`, `none`
- **Custom dimensions**: width and height props

```tsx
<Skeleton 
  variant="rectangular" 
  width={300} 
  height={200} 
  animation="wave"
/>
```

## 2. Toast Notifications

### Purpose
Provide real-time feedback for trades, system events, and user actions.

### Implementation

**Trade Notifications Hook:**
```tsx
import { useTradeNotifications } from '../hooks/useTradeNotifications';

function Trading() {
  // Automatically shows notifications for new trades from WebSocket
  useTradeNotifications();
  
  // Manual notification trigger
  const { showTradeNotification } = useTradeNotifications();
  
  const handleTrade = (trade) => {
    showTradeNotification(trade);
  };
}
```

### Toast Configuration

Global toast settings in `App.tsx`:
```tsx
<Toaster 
  theme="dark" 
  position="top-right"
  toastOptions={{
    className: 'glass border-brand-cyan/30',
    style: {
      background: 'rgba(20, 27, 61, 0.9)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(0, 217, 255, 0.2)',
      color: '#fff',
    },
  }}
/>
```

### Custom Notifications

Use Sonner's toast API for custom notifications:

```tsx
import { toast } from 'sonner';

// Success notification
toast.success('Trade executed successfully', {
  description: 'BUY BTC @ $45,000',
  duration: 5000,
});

// Error notification
toast.error('Trade failed', {
  description: 'Insufficient balance',
  duration: 5000,
});

// Info notification
toast.info('Market update', {
  description: 'New signal detected for ETH',
});
```

## 3. Dark/Light Theme Toggle

### Purpose
Allow users to switch between dark and light themes for better accessibility and preference.

### Theme Context

The theme is managed via React Context:

```tsx
import { ThemeProvider, useTheme } from '../contexts/ThemeContext';

function App() {
  return (
    <ThemeProvider>
      <YourApp />
    </ThemeProvider>
  );
}

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  
  return (
    <button onClick={toggleTheme}>
      {theme === 'dark' ? <Sun /> : <Moon />}
    </button>
  );
}
```

### Theme Persistence

- Theme preference is automatically saved to `localStorage`
- Key: `mirrorcore-theme`
- Values: `'dark'` or `'light'`

### Styling

Light theme styles are applied via CSS classes:

```css
.light-theme .glass {
  @apply bg-white/80 backdrop-blur-xl border-brand-cyan/30;
}

.light-theme .brand-card {
  @apply bg-white/90 border-brand-cyan/40;
}
```

### Custom Theme Styles

Add theme-specific styles in `index.css`:

```css
.light-theme .your-component {
  @apply bg-white text-gray-900;
}

.dark-theme .your-component {
  @apply bg-bg-secondary text-white;
}
```

## 4. Mobile Responsive Design

### Purpose
Ensure optimal user experience across all device sizes.

### Responsive Breakpoints

Using Tailwind CSS breakpoints:
- `sm`: 640px (small devices)
- `md`: 768px (tablets)
- `lg`: 1024px (laptops)
- `xl`: 1280px (desktops)

### Layout Adaptations

**Mobile-First Sidebar:**
- Collapsed by default on mobile (`< 768px`)
- Overlay mode with backdrop on mobile
- Persistent sidebar on desktop
- Automatic state management based on screen size

```tsx
const [isMobile, setIsMobile] = useState(false);

useEffect(() => {
  const checkMobile = () => {
    setIsMobile(window.innerWidth < 768);
    if (window.innerWidth >= 768) {
      setSidebarOpen(true);
    } else {
      setSidebarOpen(false);
    }
  };
  
  checkMobile();
  window.addEventListener('resize', checkMobile);
  return () => window.removeEventListener('resize', checkMobile);
}, []);
```

### Responsive Grid Layouts

```tsx
// Metric cards: 1 column on mobile, 2 on tablet, 4 on desktop
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">

// Charts: 1 column on mobile, 2 on desktop
<div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6">
```

### Responsive Spacing

```tsx
// Reduced padding on mobile
<div className="p-4 md:p-6">

// Smaller gaps on mobile
<div className="gap-4 md:gap-6">

// Responsive spacing
<div className="space-x-2 md:space-x-4">
```

### Hidden Elements on Mobile

```tsx
// Hide on small screens
<button className="hidden sm:block">

// Show only on mobile
<button className="block sm:hidden">
```

### Touch-Friendly Interactions

- Larger tap targets (minimum 44px)
- Mobile overlay for sidebar dismissal
- Swipe-friendly navigation
- Optimized chart interactions for touch

## Best Practices

### 1. Loading States
- Always show loading skeletons for async data
- Match skeleton layout to actual content
- Use pulse animation for unknown duration

### 2. Notifications
- Keep messages concise and actionable
- Use appropriate toast types (success, error, info)
- Set reasonable durations (3-5 seconds)
- Position consistently (top-right recommended)

### 3. Theming
- Test all components in both themes
- Ensure sufficient contrast in light mode
- Preserve brand colors across themes
- Use semantic color tokens

### 4. Responsive Design
- Design mobile-first
- Test on real devices when possible
- Consider touch targets and gestures
- Optimize chart/graph readability on small screens

## Integration Checklist

When adding new pages or components:

- [ ] Add loading skeleton for async data
- [ ] Implement appropriate toast notifications
- [ ] Test in both dark and light themes
- [ ] Verify mobile responsiveness
- [ ] Check touch interactions on mobile
- [ ] Ensure accessibility (ARIA labels, keyboard nav)
- [ ] Test on multiple screen sizes

## Examples

See the Dashboard page (`src/pages/Dashboard.tsx`) for a complete implementation example showcasing all UX enhancements.
