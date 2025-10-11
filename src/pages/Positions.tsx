
import React from 'react';
import { useAPI } from '../hooks/useAPI';
import { TrendingUp, TrendingDown, DollarSign, Target, AlertCircle } from 'lucide-react';

const Positions: React.FC = () => {
  const { data: positionsData } = useAPI('/api/positions/active');

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Open Positions</h1>
        <p className="page-subtitle">Active portfolio positions and P&L tracking</p>
      </div>

      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
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
      </div>

      {/* Positions Table */}
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
    </div>
  );
};

export default Positions;
