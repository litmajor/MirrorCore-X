import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Trading from './pages/Trading';
import Analytics from './pages/Analytics';
import TechnicalAnalysis from './pages/TechnicalAnalysis';
import Strategies from './pages/Strategies';
import RiskManagement from './pages/RiskManagement';
import Settings from './pages/Settings';
import Oracle from './pages/Oracle';
import Optimization from './pages/Optimization';
import RLAgent from './pages/RLAgent'; // Assuming RLAgent component is in './pages/RLAgent'
import Layout from './components/Layout';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/trading" element={<Trading />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/technical" element={<TechnicalAnalysis />} />
          <Route path="/strategies" element={<Strategies />} />
          <Route path="/risk" element={<RiskManagement />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/oracle" element={<Oracle />} />
          <Route path="/optimization" element={<Optimization />} />
          <Route path="/rl-agent" element={<RLAgent />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;