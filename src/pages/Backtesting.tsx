
import React from 'react';
import { useAPI } from '../hooks/useAPI';
import { BarChart3, TrendingUp, TrendingDown, Activity, Calendar, DollarSign } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const Backtesting: React.FC = () => {
  const { data: backtestResults } = useAPI('/api/backtest/results');

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Backtesting Results</h1>
        <p className="page-subtitle">Comprehensive strategy performance analysis</p>
      </div>

      {backtestResults?.error ? (
        <div className="chart-container">
          <div className="flex items-center justify-center p-8 text-txt-secondary">
            <Activity className="w-5 h-5 mr-2" />
            {backtestResults.error}
          </div>
        </div>
      ) : (
        <>
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="metric-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-txt-secondary text-sm font-medium">Total Return</h3>
                <TrendingUp className="w-5 h-5 text-success" />
              </div>
              <p className={`text-2xl font-bold ${(backtestResults?.total_return || 0) >= 0 ? 'text-success' : 'text-error'}`}>
                {(backtestResults?.total_return || 0).toFixed(2)}%
              </p>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-txt-secondary text-sm font-medium">Sharpe Ratio</h3>
                <BarChart3 className="w-5 h-5 text-brand-cyan" />
              </div>
              <p className="text-2xl font-bold text-white">
                {(backtestResults?.sharpe_ratio || 0).toFixed(2)}
              </p>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-txt-secondary text-sm font-medium">Max Drawdown</h3>
                <TrendingDown className="w-5 h-5 text-error" />
              </div>
              <p className="text-2xl font-bold text-error">
                {(backtestResults?.max_drawdown || 0).toFixed(2)}%
              </p>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-txt-secondary text-sm font-medium">Win Rate</h3>
                <DollarSign className="w-5 h-5 text-brand-purple" />
              </div>
              <p className="text-2xl font-bold text-white">
                {(backtestResults?.win_rate || 0).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Backtest Report */}
          <div className="chart-container">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Full Report</h3>
              <div className="flex items-center space-x-2 text-txt-secondary text-sm">
                <Calendar className="w-4 h-4" />
                <span>{backtestResults?.timestamp ? new Date(backtestResults.timestamp).toLocaleString() : 'N/A'}</span>
              </div>
            </div>
            <div className="bg-bg-tertiary rounded-lg p-4 overflow-auto max-h-96">
              <pre className="text-txt-secondary text-sm font-mono whitespace-pre-wrap">
                {backtestResults?.report_text || 'No backtest report available'}
              </pre>
            </div>
          </div>

          {/* Regime Performance */}
          <div className="chart-container">
            <h3 className="text-lg font-semibold text-white mb-4">Performance by Market Regime</h3>
            <p className="text-txt-secondary text-sm">
              Detailed regime analysis available in the full report above
            </p>
          </div>
        </>
      )}
    </div>
  );
};

export default Backtesting;
