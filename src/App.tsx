import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'sonner';
import { ThemeProvider } from './contexts/ThemeContext';
import Dashboard from './pages/Dashboard';
import Trading from './pages/Trading';
import Analytics from './pages/Analytics';
import TechnicalAnalysis from './pages/TechnicalAnalysis';
import Strategies from './pages/Strategies';
import RiskManagement from './pages/RiskManagement';
import Settings from './pages/Settings';
import Oracle from './pages/Oracle';
import Optimization from './pages/Optimization';
import RLAgent from './pages/RLAgent';
import Backtesting from './pages/Backtesting';
import AgentMonitor from './pages/AgentMonitor';
import Audit from './pages/Audit';
import Positions from './pages/Positions';
import TradeHistory from './pages/TradeHistory';
import UXDemo from './pages/UXDemo';
import MathOptimizer from './pages/MathOptimizer';
import Layout from './components/Layout';
import ErrorBoundary from './components/ErrorBoundary';
import { TrendingUp, Activity, BarChart3, Settings as SettingsIcon, Shield, History, Target, Brain, Sparkles, Layers, TestTube, Calculator } from 'lucide-react';

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/trading" element={<Trading />} />
              <Route path="/positions" element={<Positions />} />
              <Route path="/trade-history" element={<TradeHistory />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/technical" element={<TechnicalAnalysis />} />
              <Route path="/strategies" element={<Strategies />} />
              <Route path="/risk" element={<RiskManagement />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/oracle" element={<Oracle />} />
              <Route path="/math-optimizer" element={<MathOptimizer />} />
              <Route path="/ux-demo" element={<UXDemo />} />
              <Route path="/optimization" element={<Optimization />} />
              <Route path="/rl-agent" element={<RLAgent />} />
              <Route path="/backtesting" element={<Backtesting />} />
              <Route path="/agent-monitor" element={<AgentMonitor />} />
              <Route path="/audit" element={<Audit />} />
            </Routes>
          </Layout>
        </Router>
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
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;