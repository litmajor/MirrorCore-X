
import React, { useState } from 'react';
import { useAPI } from '../hooks/useAPI';
import { BarChart3, TrendingUp, TrendingDown, Activity, Calendar, DollarSign, Trophy, Award, Play } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';

const Backtesting: React.FC = () => {
  const { data: backtestResults } = useAPI('/api/backtest/results');
  const { data: comparisonResults, refetch: refetchComparison } = useAPI('/api/backtest/comparison');
  const [isRunningComparison, setIsRunningComparison] = useState(false);

  const runComparison = async () => {
    setIsRunningComparison(true);
    try {
      const response = await fetch('/api/backtest/run-comparison', { method: 'POST' });
      await response.json();
      await refetchComparison();
    } catch (error) {
      console.error('Comparison failed:', error);
    } finally {
      setIsRunningComparison(false);
    }
  };

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

          {/* Strategy Comparison Section */}
          <div className="chart-container">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-white flex items-center">
                <Trophy className="w-6 h-6 mr-2 text-brand-cyan" />
                Strategy Head-to-Head Comparison
              </h3>
              <button
                onClick={runComparison}
                disabled={isRunningComparison}
                className="btn-primary flex items-center space-x-2"
              >
                <Play className="w-4 h-4" />
                <span>{isRunningComparison ? 'Running...' : 'Run Comparison'}</span>
              </button>
            </div>

            {comparisonResults && !comparisonResults.error ? (
              <div className="space-y-6">
                {/* Summary Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="metric-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-txt-secondary text-sm">Strategies Tested</span>
                      <BarChart3 className="w-5 h-5 text-brand-purple" />
                    </div>
                    <p className="text-2xl font-bold text-white">
                      {comparisonResults.summary.total_strategies}
                    </p>
                  </div>
                  <div className="metric-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-txt-secondary text-sm">Best Strategy</span>
                      <Award className="w-5 h-5 text-brand-cyan" />
                    </div>
                    <p className="text-lg font-bold text-white">
                      {comparisonResults.summary.best_strategy}
                    </p>
                  </div>
                  <div className="metric-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-txt-secondary text-sm">Best Sharpe Ratio</span>
                      <TrendingUp className="w-5 h-5 text-success" />
                    </div>
                    <p className="text-2xl font-bold text-white">
                      {comparisonResults.summary.best_sharpe.toFixed(2)}
                    </p>
                  </div>
                </div>

                {/* Top 5 Performers */}
                <div>
                  <h4 className="text-lg font-semibold text-white mb-4">Top 5 Performers</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-slate-700">
                          <th className="text-left py-3 px-4 text-txt-secondary font-medium">Rank</th>
                          <th className="text-left py-3 px-4 text-txt-secondary font-medium">Strategy</th>
                          <th className="text-right py-3 px-4 text-txt-secondary font-medium">Total Return</th>
                          <th className="text-right py-3 px-4 text-txt-secondary font-medium">Sharpe</th>
                          <th className="text-right py-3 px-4 text-txt-secondary font-medium">Max DD</th>
                          <th className="text-right py-3 px-4 text-txt-secondary font-medium">Win Rate</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonResults.rankings.by_sharpe.slice(0, 5).map((strategy: any) => (
                          <tr key={strategy.rank} className="border-b border-slate-700/50 hover:bg-slate-800/30">
                            <td className="py-3 px-4">
                              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                                strategy.rank === 1 ? 'bg-yellow-500/20 text-yellow-400' :
                                strategy.rank === 2 ? 'bg-slate-400/20 text-slate-300' :
                                strategy.rank === 3 ? 'bg-orange-600/20 text-orange-400' :
                                'bg-slate-700 text-txt-secondary'
                              }`}>
                                {strategy.rank}
                              </div>
                            </td>
                            <td className="py-3 px-4 font-medium text-white">{strategy.strategy}</td>
                            <td className={`py-3 px-4 text-right font-semibold ${
                              strategy.total_return >= 0 ? 'text-success' : 'text-error'
                            }`}>
                              {strategy.total_return.toFixed(2)}%
                            </td>
                            <td className="py-3 px-4 text-right font-semibold text-white">
                              {strategy.sharpe_ratio.toFixed(2)}
                            </td>
                            <td className="py-3 px-4 text-right font-semibold text-error">
                              {strategy.max_drawdown.toFixed(2)}%
                            </td>
                            <td className="py-3 px-4 text-right font-semibold text-white">
                              {strategy.win_rate.toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Performance Chart */}
                <div>
                  <h4 className="text-lg font-semibold text-white mb-4">Strategy Performance Comparison</h4>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={comparisonResults.rankings.by_sharpe.slice(0, 10)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="strategy" stroke="#9CA3AF" angle={-45} textAnchor="end" height={100} />
                        <YAxis stroke="#9CA3AF" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1E293B',
                            border: '1px solid #334155',
                            borderRadius: '8px',
                          }}
                        />
                        <Bar dataKey="sharpe_ratio" fill="#06b6d4" name="Sharpe Ratio" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* All Strategies Ranking */}
                <div>
                  <h4 className="text-lg font-semibold text-white mb-4">Complete Rankings</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {comparisonResults.all_results.map((result: any, idx: number) => (
                      <div key={result.strategy} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
                        <div className="flex items-center justify-between mb-2">
                          <h5 className="font-semibold text-white">{result.strategy}</h5>
                          <span className="text-sm text-txt-secondary">#{idx + 1}</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-txt-secondary">Return:</span>
                            <span className={result.total_return >= 0 ? 'text-success' : 'text-error'}>
                              {result.total_return.toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-txt-secondary">Sharpe:</span>
                            <span className="text-white">{result.sharpe_ratio.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-txt-secondary">Max DD:</span>
                            <span className="text-error">{result.max_drawdown.toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-txt-secondary">Trades:</span>
                            <span className="text-white">{result.total_trades}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Trophy className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                <p className="text-txt-secondary mb-4">
                  Compare all 19 strategies head-to-head with realistic backtesting
                </p>
                <button
                  onClick={runComparison}
                  disabled={isRunningComparison}
                  className="btn-primary"
                >
                  {isRunningComparison ? 'Running Comparison...' : 'Run Strategy Comparison'}
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default Backtesting;
