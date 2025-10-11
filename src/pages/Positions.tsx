
import React from 'react';
import { useAPI } from '../hooks/useAPI';
import { TrendingUp, TrendingDown, DollarSign, Target, AlertCircle, PieChart as PieChartIcon } from 'lucide-react';
import { LineChart, Line, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { MetricCardSkeleton, ChartSkeleton, TableRowSkeleton } from '../components/Skeleton';

const Positions: React.FC = () => {
  const { data: positionsData, loading, error } = useAPI('/api/positions/active');
  const { data: performanceData } = useAPI('/api/performance/summary');

  // Calculate portfolio allocation data
  const allocationData = positionsData?.positions?.map((p: any) => ({
    name: p.symbol,
    value: Math.abs(p.size * p.current_price)
  })) || [];

  // Calculate P&L distribution
  const pnlDistribution = positionsData?.positions?.map((p: any) => ({
    symbol: p.symbol,
    pnl: p.pnl,
    pnl_percent: p.pnl_percent
  })) || [];

  const COLORS = ['#00D9FF', '#7B2FFF', '#00FF88', '#FFB800', '#FF3366', '#00FFC8', '#FF6B9D'];

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Open Positions</h1>
        <p className="page-subtitle">Active portfolio positions and P&L tracking</p>
      </div>

      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
        {loading ? (
          <>
            <MetricCardSkeleton />
            <MetricCardSkeleton />
            <MetricCardSkeleton />
            <MetricCardSkeleton />
          </>
        ) : error ? (
          <div className="col-span-full brand-card text-center py-12">
            <PieChartIcon className="w-16 h-16 text-txt-secondary mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">Unable to Load Positions</h3>
            <p className="text-txt-secondary">Start the backend to view your portfolio positions.</p>
          </div>
        ) : (
          <>
            <div className="metric-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-txt-secondary text-sm font-medium">Total Value</h3>
                <DollarSign className="w-5 h-5 text-brand-cyan" />
              </div>
              <p className="text-2xl font-bold text-white">
                ${positionsData?.total_value?.toFixed(2) || '0.00'}
              </p>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-txt-secondary text-sm font-medium">Unrealized P&L</h3>
                <TrendingUp className="w-5 h-5 text-success" />
              </div>
              <p className={`text-2xl font-bold ${(positionsData?.unrealized_pnl || 0) >= 0 ? 'text-success' : 'text-error'}`}>
                ${positionsData?.unrealized_pnl?.toFixed(2) || '0.00'}
              </p>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-txt-secondary text-sm font-medium">Open Positions</h3>
                <Target className="w-5 h-5 text-brand-purple" />
              </div>
              <p className="text-2xl font-bold text-white">
                {positionsData?.positions?.length || 0}
              </p>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-txt-secondary text-sm font-medium">Avg Return</h3>
                <AlertCircle className="w-5 h-5 text-warning" />
              </div>
              <p className={`text-2xl font-bold ${(positionsData?.avg_return || 0) >= 0 ? 'text-success' : 'text-error'}`}>
                {positionsData?.avg_return?.toFixed(2) || '0.00'}%
              </p>
            </div>
          </>
        )}
      </div>

      {/* Portfolio Visualizations */}
      {!error && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6">
        {/* Portfolio Allocation */}
        <div className="chart-container">
          <h3 className="text-lg font-semibold text-white mb-4">Portfolio Allocation</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={allocationData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.name}: ${(entry.value / positionsData?.total_value * 100).toFixed(1)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {allocationData.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #00D9FF',
                  borderRadius: '8px'
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* P&L Distribution */}
        <div className="chart-container">
          <h3 className="text-lg font-semibold text-white mb-4">P&L by Position</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={pnlDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="symbol" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #00D9FF',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="pnl" fill="#00D9FF">
                {pnlDistribution.map((entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={entry.pnl >= 0 ? '#00FF88' : '#FF3366'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        </div>
      )}

      {/* Positions Table */}
      {!error && (
        <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4">Active Positions</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="border-b border-brand-cyan/20">
              <tr>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Symbol</th>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Side</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">Size</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">Entry Price</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">Current Price</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">P&L</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">P&L %</th>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Strategy</th>
              </tr>
            </thead>
            <tbody>
              {positionsData?.positions?.map((position: any, idx: number) => (
                <tr key={idx} className="border-b border-brand-cyan/10 hover:bg-bg-tertiary/50">
                  <td className="py-3 px-4 text-white font-medium">{position.symbol}</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      position.side === 'long' ? 'bg-success/20 text-success' : 'bg-error/20 text-error'
                    }`}>
                      {position.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right text-white">{position.size?.toFixed(4)}</td>
                  <td className="py-3 px-4 text-right text-white">${position.entry_price?.toFixed(2)}</td>
                  <td className="py-3 px-4 text-right text-white">${position.current_price?.toFixed(2)}</td>
                  <td className={`py-3 px-4 text-right font-bold ${position.pnl >= 0 ? 'text-success' : 'text-error'}`}>
                    ${position.pnl?.toFixed(2)}
                  </td>
                  <td className={`py-3 px-4 text-right font-bold ${position.pnl_percent >= 0 ? 'text-success' : 'text-error'}`}>
                    {position.pnl_percent?.toFixed(2)}%
                  </td>
                  <td className="py-3 px-4 text-txt-secondary text-sm">{position.strategy || 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {(!positionsData?.positions || positionsData.positions.length === 0) && (
            <div className="text-center py-8 text-txt-secondary">
              No open positions
            </div>
          )}
        </div>
        </div>
      )}
    </div>
  );
};

export default Positions;
