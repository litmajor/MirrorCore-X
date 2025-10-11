import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, TrendingUp, BarChart3, Layers, Shield, 
  Settings, Menu, X, Bell, User, Brain 
} from 'lucide-react';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const location = useLocation();

  const navItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/trading', icon: TrendingUp, label: 'Trading' },
    { path: '/analytics', icon: BarChart3, label: 'Analytics' },
    { path: '/strategies', icon: Layers, label: 'Strategies' },
    { path: '/risk', icon: Shield, label: 'Risk Management' },
    { path: '/settings', icon: Settings, label: 'Settings' },
    { path: '/rl-agent', icon: Brain, label: 'RL Agent' },
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

          <div className="flex items-center space-x-4">
            <button className="p-2 hover:bg-bg-surface rounded-lg transition-colors relative">
              <Bell className="w-5 h-5 text-brand-cyan" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-error rounded-full"></span>
            </button>
            <button className="p-2 hover:bg-bg-surface rounded-lg transition-colors">
              <User className="w-5 h-5 text-brand-cyan" />
            </button>
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div
        className={`fixed left-0 top-16 bottom-0 w-64 glass border-r border-brand-cyan/20 transition-transform duration-300 z-40 custom-scrollbar ${
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
          sidebarOpen ? 'pl-64' : 'pl-0'
        }`}
      >
        <div className="p-6 custom-scrollbar">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Layout;