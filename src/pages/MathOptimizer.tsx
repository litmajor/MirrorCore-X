
import React, { useState, useEffect } from 'react';
import { LineChart, Line, ScatterChart, Scatter, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Settings, TrendingUp, Shield, Zap, AlertCircle, Calculator } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';

const MathOptimizer: React.FC = () => {
  const [lambdaRisk, setLambdaRisk] = useState(100);
  const [etaTurnover, setEtaTurnover] = useState(0.05);
  const [maxWeight, setMaxWeight] = useState(0.25);
  const [regime, setRegime] = useState('trending');
  const [useShrinkage, setUseShrinkage] = useState(true);
  const [useResampling, setUseResampling] = useState(false);

  const { data: optimizerData, loading } = useAPI('/api/optimizer/weights', {
    lambda: lambdaRisk,
    eta: etaTurnover,
    max_weight: maxWeight,
    regime: regime,
    shrinkage: useShrinkage,
    resampling: useResampling
  });

  const getColorForCategory = (category: string) => {
    const colors: Record<string, string> = {
      trend: '#3b82f6',
      support: '#10b981',
      reversion: '#8b5cf6',
      breakout: '#ef4444',
      adaptive: '#f59e0b',
      arbitrage: '#14b8a6',
      ml: '#ec4899',
      hybrid: '#06b6d4',
      pattern: '#6366f1',
      'multi-asset': '#84cc16',
      hft: '#f97316',
      derivatives: '#a855f7',
      information: '#0ea5e9',
      probabilistic: '#22d3ee',
      causal: '#64748b'
    };
    return colors[category] || '#94a3b8';
  };

  const weightData = optimizerData?.weights?.map((w: any) => ({
    name: w.name.replace(/_/g, ' '),
    weight: w.weight * 100,
    category: w.category
  })).sort((a: any, b: any) => b.weight - a.weight).slice(0, 10) || [];

  return (
    <div className="space-y-6">
      <div className="page-header">
        <div className="flex items-center gap-3">
          <Calculator className="w-10 h-10 text-brand-cyan" />
          <div>
            <h1 className="page-title">Mathematical Portfolio Optimizer</h1>
            <p className="page-subtitle">Quadratic Programming with Ledoit-Wolf Shrinkage</p>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card bg-gradient-to-br from-brand-cyan/20 to-brand-purple/20 border-brand-cyan/30">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-txt-secondary">Expected Return</span>
            <TrendingUp className="w-5 h-5 text-brand-cyan" />
          </div>
          <div className="text-3xl font-bold text-brand-cyan">
            {(optimizerData?.expected_return * 100 || 0).toFixed(3)}%
          </div>
        </div>

        <div className="metric-card bg-gradient-to-br from-brand-purple/20 to-accent/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-txt-secondary">Expected Vol</span>
            <Shield className="w-5 h-5 text-brand-purple" />
          </div>
          <div className="text-3xl font-bold text-brand-purple">
            {(optimizerData?.expected_volatility * 100 || 0).toFixed(2)}%
          </div>
        </div>

        <div className="metric-card bg-gradient-to-br from-success/20 to-brand-cyan/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-txt-secondary">Sharpe Ratio</span>
            <Zap className="w-5 h-5 text-success" />
          </div>
          <div className="text-3xl font-bold text-success">
            {optimizerData?.sharpe_ratio?.toFixed(3) || '0.000'}
          </div>
        </div>

        <div className="metric-card bg-gradient-to-br from-warning/20 to-error/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-txt-secondary">Turnover</span>
            <AlertCircle className="w-5 h-5 text-warning" />
          </div>
          <div className="text-3xl font-bold text-warning">
            {(optimizerData?.turnover * 100 || 0).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="brand-card">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5 text-brand-cyan" />
          <h3 className="text-lg font-semibold text-white">Optimization Parameters</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-sm font-medium mb-2 text-txt-secondary">
              λ (Risk Aversion): <span className="text-brand-cyan">{lambdaRisk}</span>
            </label>
            <input
              type="range"
              min="10"
              max="500"
              step="10"
              value={lambdaRisk}
              onChange={(e) => setLambdaRisk(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-txt-muted mt-1">
              <span>Aggressive</span>
              <span>Conservative</span>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-txt-secondary">
              η (Turnover Penalty): <span className="text-brand-cyan">{etaTurnover.toFixed(3)}</span>
            </label>
            <input
              type="range"
              min="0.01"
              max="0.20"
              step="0.01"
              value={etaTurnover}
              onChange={(e) => setEtaTurnover(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-txt-secondary">
              Max Weight: <span className="text-brand-cyan">{(maxWeight * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0.10"
              max="0.50"
              step="0.05"
              value={maxWeight}
              onChange={(e) => setMaxWeight(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-txt-secondary">Market Regime</label>
            <select
              value={regime}
              onChange={(e) => setRegime(e.target.value)}
              className="w-full bg-bg-tertiary border border-accent-dark rounded px-3 py-2 text-white"
            >
              <option value="trending">Trending</option>
              <option value="ranging">Range-Bound</option>
              <option value="volatile">High Volatility</option>
              <option value="mixed">Mixed/Uncertain</option>
            </select>
          </div>

          <div className="flex flex-col gap-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useShrinkage}
                onChange={(e) => setUseShrinkage(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-txt-secondary">Ledoit-Wolf Shrinkage</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={useResampling}
                onChange={(e) => setUseResampling(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm text-txt-secondary">Bootstrap Resampling</span>
            </label>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="chart-container">
          <h3 className="text-lg font-semibold text-white mb-4">Top 10 Strategy Weights</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={weightData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" stroke="#94a3b8" />
              <YAxis dataKey="name" type="category" width={150} stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="weight" radius={[0, 8, 8, 0]}>
                {weightData.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={getColorForCategory(entry.category)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="brand-card">
          <h3 className="text-lg font-semibold text-white mb-3">Optimization Formula</h3>
          <div className="bg-bg-tertiary p-4 rounded-lg font-mono text-sm overflow-x-auto">
            <div className="text-brand-cyan mb-3">Objective Function:</div>
            <div className="text-txt-secondary mb-4">
              max<sub>w</sub> w'μ - (λ/2)w'Σw - η||w - w<sub>prev</sub>||²
            </div>
            
            <div className="text-brand-cyan mb-3">Subject to:</div>
            <div className="text-txt-secondary space-y-1">
              <div>• Σw<sub>i</sub> = 1 (fully invested)</div>
              <div>• 0 ≤ w<sub>i</sub> ≤ {maxWeight.toFixed(2)} (bounds)</div>
              <div>• λ = {lambdaRisk} (risk aversion)</div>
              <div>• η = {etaTurnover.toFixed(3)} (turnover penalty)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MathOptimizer;
