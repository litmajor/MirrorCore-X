
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Trading from './pages/Trading';
import Analytics from './pages/Analytics';
import Strategies from './pages/Strategies';
import RiskManagement from './pages/RiskManagement';
import Settings from './pages/Settings';
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
          <Route path="/strategies" element={<Strategies />} />
          <Route path="/risk" element={<RiskManagement />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
