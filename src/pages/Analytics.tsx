
import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, DollarSign, Target, AlertTriangle } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';

const Analytics: React.FC = () => {
  const { data: performance } = useAPI('/api/performance/summary');
  const { data: strategies } = useAPI('/api/strategies');

  const COLORS = ['#00D9FF', '#7B2FFF', '#00FF88', '#FFB800', '#FF3366'];

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Performance Analytics</h1>
        <p className="page-subtitle">Comprehensive trading performance metrics</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Total P&L</h3>
            <DollarSign className="w-5 h-5 text-success" />
          </div>
          <p className="text-3xl font-bold text-white">${performance?.total_pnl?.toFixed(2) || '0.00'}</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Win Rate</h3>
            <Target className="w-5 h-5 text-brand-cyan" />
          </div>
          <p className="text-3xl font-bold text-white">{performance?.win_rate?.toFixed(1) || '0'}%</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Sharpe Ratio</h3>
            <TrendingUp className="w-5 h-5 text-brand-purple" />
          </div>
          <p className="text-3xl font-bold text-white">{performance?.sharpe_ratio?.toFixed(2) || '0.00'}</p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-txt-secondary text-sm font-medium">Max Drawdown</h3>
            <AlertTriangle className="w-5 h-5 text-error" />
          </div>
          <p className="text-3xl font-bold text-white">{performance?.max_drawdown?.toFixed(1) || '0'}%</p>
        </div>
      </div>

      {/* Equity Curve */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4">Portfolio Equity Curve</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={performance?.equity_curve || []}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="date" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1e293b', 
                border: '1px solid #00D9FF',
                borderRadius: '8px'
              }}
            />
            <Line type="monotone" dataKey="equity" stroke="#00D9FF" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown Chart */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4">Drawdown Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={performance?.drawdown_curve || []}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="date" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1e293b', 
                border: '1px solid #00D9FF',
                borderRadius: '8px'
              }}
            />
            <Line type="monotone" dataKey="drawdown" stroke="#FF3366" strokeWidth={2} fill="#FF3366" fillOpacity={0.3} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Strategy Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="chart-container">
          <h3 className="text-lg font-semibold text-white mb-4">Strategy P&L Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={strategies?.strategies?.map((s: any) => ({ name: s.name, value: s.pnl })) || []}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.name}: $${entry.value}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {strategies?.strategies?.map((_: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3 className="text-lg font-semibold text-white mb-4">Win Rate by Strategy</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={strategies?.strategies || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #334155',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="win_rate" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
