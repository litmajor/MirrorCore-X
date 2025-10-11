import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, TrendingUp, BarChart3, Layers, Shield, 
  Settings, Menu, X, Bell, User, Brain, Target, History, Sun, Moon 
} from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

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

  const navItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/trading', icon: TrendingUp, label: 'Trading' },
    { path: '/positions', icon: Target, label: 'Positions' },
    { path: '/trade-history', icon: History, label: 'Trade History' },
    { path: '/analytics', icon: BarChart3, label: 'Analytics' },
    { path: '/strategies', icon: Layers, label: 'Strategies' },
    { path: '/risk', icon: Shield, label: 'Risk Management' },
    { path: '/backtesting', icon: BarChart3, label: 'Backtesting' },
    { path: '/agent-monitor', icon: Brain, label: 'Agent Monitor' },
    { path: '/rl-agent', icon: Brain, label: 'RL Agent' },
    { path: '/audit', icon: Shield, label: 'Audit Trail' },
    { path: '/settings', icon: Settings, label: 'Settings' },
  ];

  return (
    <div className="min-h-screen bg-gradient-secondary">
      {/* Top Navigation Bar */}
      <div className="fixed top-0 left-0 right-0 h-16 glass border-b border-brand-cyan/20 z-50">
        <div className="flex items-center justify-between h-full px-6">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-bg-surface rounded-lg transition-colors"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <h1 className="text-xl font-display font-bold gradient-text neon-text">
              MIRRORCORE-X
            </h1>
          </div>

          <div className="flex items-center space-x-2 md:space-x-4">
            <button
              onClick={toggleTheme}
              className="p-2 hover:bg-bg-surface rounded-lg transition-colors"
              aria-label="Toggle theme"
            >
              {theme === 'dark' ? (
                <Sun className="w-5 h-5 text-brand-cyan" />
              ) : (
                <Moon className="w-5 h-5 text-brand-cyan" />
              )}
            </button>
            <button className="p-2 hover:bg-bg-surface rounded-lg transition-colors relative hidden sm:block">
              <Bell className="w-5 h-5 text-brand-cyan" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-error rounded-full"></span>
            </button>
            <button className="p-2 hover:bg-bg-surface rounded-lg transition-colors hidden sm:block">
              <User className="w-5 h-5 text-brand-cyan" />
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 top-16"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={`fixed left-0 top-16 bottom-0 w-64 glass border-r border-brand-cyan/20 transition-transform duration-300 ${
          isMobile ? 'z-40' : 'z-30'
        } custom-scrollbar overflow-y-auto ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <nav className="p-4 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;

            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => isMobile && setSidebarOpen(false)}
                className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
                  isActive
                    ? 'nav-active'
                    : 'nav-inactive'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Main Content */}
      <div
        className={`pt-16 transition-all duration-300 ${
          sidebarOpen && !isMobile ? 'md:pl-64' : 'pl-0'
        }`}
      >
        <div className="p-4 md:p-6 custom-scrollbar">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Layout;