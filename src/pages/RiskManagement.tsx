
import React from 'react';
import { Shield, AlertTriangle, TrendingDown, Activity } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const RiskManagement: React.FC = () => {
  const { data: riskData } = useAPI('/api/risk/analysis');

  const getRiskLevel = (value: number, threshold: number) => {
    if (value > threshold) return 'text-error';
    if (value > threshold * 0.7) return 'text-warning';
    return 'text-success';
  };

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Risk Management</h1>
        <p className="page-subtitle">Monitor and control trading risk exposure</p>
      </div>

      {/* Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">VaR (95%)</h3>
            <AlertTriangle className="w-5 h-5 text-warning" />
          </div>
          <p className={`text-3xl font-bold ${getRiskLevel(riskData?.var_95 || 0, 3000)}`}>
            ${riskData?.var_95?.toFixed(2) || '0.00'}
          </p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Position Concentration</h3>
            <Activity className="w-5 h-5 text-brand-cyan" />
          </div>
          <p className={`text-3xl font-bold ${getRiskLevel(riskData?.position_concentration || 0, 0.5)}`}>
            {((riskData?.position_concentration || 0) * 100).toFixed(1)}%
          </p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Correlation Risk</h3>
            <TrendingDown className="w-5 h-5 text-brand-purple" />
          </div>
          <p className={`text-3xl font-bold ${getRiskLevel(riskData?.correlation_risk || 0, 0.6)}`}>
            {((riskData?.correlation_risk || 0) * 100).toFixed(1)}%
          </p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Leverage</h3>
            <Shield className="w-5 h-5 text-success" />
          </div>
          <p className={`text-3xl font-bold ${getRiskLevel(riskData?.leverage || 0, 3)}`}>
            {riskData?.leverage?.toFixed(1)}x
          </p>
        </div>
      </div>

      {/* Risk Limits */}
      <div className="brand-card">
        <h3 className="text-lg font-semibold text-white mb-4">Risk Limits & Controls</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-bg-tertiary rounded-lg">
            <div>
              <p className="text-white font-medium">Max Drawdown Limit</p>
              <p className="text-txt-secondary text-sm">Circuit breaker triggers at -15%</p>
            </div>
            <span className="text-success font-bold">Active</span>
          </div>
          <div className="flex items-center justify-between p-4 bg-bg-tertiary rounded-lg">
            <div>
              <p className="text-white font-medium">Position Size Limit</p>
              <p className="text-txt-secondary text-sm">Max 30% per position</p>
            </div>
            <span className="text-success font-bold">Active</span>
          </div>
          <div className="flex items-center justify-between p-4 bg-bg-tertiary rounded-lg">
            <div>
              <p className="text-white font-medium">Daily Loss Limit</p>
              <p className="text-txt-secondary text-sm">Stop trading at -5% daily loss</p>
            </div>
            <span className="text-success font-bold">Active</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskManagement;
