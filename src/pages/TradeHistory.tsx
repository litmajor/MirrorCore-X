
import React, { useState } from 'react';
import { useAPI } from '../hooks/useAPI';
import { History, Filter, Download, TrendingUp, TrendingDown } from 'lucide-react';

const TradeHistory: React.FC = () => {
  const { data: tradesData } = useAPI('/api/trades/history');
  const [filterType, setFilterType] = useState<string>('all');

  const filteredTrades = tradesData?.trades?.filter((trade: any) => {
    if (filterType === 'all') return true;
    if (filterType === 'winning') return trade.pnl > 0;
    if (filterType === 'losing') return trade.pnl < 0;
    return true;
  }) || [];

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Trade History</h1>
        <p className="page-subtitle">Complete trading activity log</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="metric-card">
          <h3 className="text-txt-secondary text-sm mb-2">Total Trades</h3>
          <p className="text-2xl font-bold text-white">{tradesData?.total_trades || 0}</p>
        </div>
        <div className="metric-card">
          <h3 className="text-txt-secondary text-sm mb-2">Winning Trades</h3>
          <p className="text-2xl font-bold text-success">{tradesData?.winning_trades || 0}</p>
        </div>
        <div className="metric-card">
          <h3 className="text-txt-secondary text-sm mb-2">Losing Trades</h3>
          <p className="text-2xl font-bold text-error">{tradesData?.losing_trades || 0}</p>
        </div>
        <div className="metric-card">
          <h3 className="text-txt-secondary text-sm mb-2">Avg Trade Duration</h3>
          <p className="text-2xl font-bold text-white">{tradesData?.avg_duration || '0h'}</p>
        </div>
      </div>

      {/* Filters */}
      <div className="chart-container">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-txt-secondary" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-4 py-2 bg-bg-tertiary border border-brand-cyan/20 rounded-lg text-white focus:outline-none focus:border-brand-cyan"
            >
              <option value="all">All Trades</option>
              <option value="winning">Winning Trades</option>
              <option value="losing">Losing Trades</option>
            </select>
          </div>
          <button className="btn-primary flex items-center space-x-2">
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </button>
        </div>

        {/* Trades Table */}
        <div className="overflow-x-auto max-h-[600px]">
          <table className="w-full">
            <thead className="sticky top-0 bg-bg-secondary border-b border-brand-cyan/20">
              <tr>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Date</th>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Symbol</th>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Side</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">Entry</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">Exit</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">Size</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">P&L</th>
                <th className="text-right py-3 px-4 text-txt-secondary text-sm font-medium">Fees</th>
                <th className="text-left py-3 px-4 text-txt-secondary text-sm font-medium">Strategy</th>
              </tr>
            </thead>
            <tbody>
              {filteredTrades.map((trade: any, idx: number) => (
                <tr key={idx} className="border-b border-brand-cyan/10 hover:bg-bg-tertiary/50">
                  <td className="py-3 px-4 text-txt-secondary text-sm">
                    {new Date(trade.timestamp).toLocaleDateString()}
                  </td>
                  <td className="py-3 px-4 text-white font-medium">{trade.symbol}</td>
                  <td className="py-3 px-4">
                    <div className="flex items-center space-x-1">
                      {trade.side === 'buy' ? (
                        <TrendingUp className="w-4 h-4 text-success" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-error" />
                      )}
                      <span className={trade.side === 'buy' ? 'text-success' : 'text-error'}>
                        {trade.side.toUpperCase()}
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right text-white">${trade.entry_price?.toFixed(2)}</td>
                  <td className="py-3 px-4 text-right text-white">${trade.exit_price?.toFixed(2)}</td>
                  <td className="py-3 px-4 text-right text-white">{trade.size?.toFixed(4)}</td>
                  <td className={`py-3 px-4 text-right font-bold ${trade.pnl >= 0 ? 'text-success' : 'text-error'}`}>
                    ${trade.pnl?.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right text-txt-secondary text-sm">
                    ${trade.fees?.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-txt-secondary text-sm">{trade.strategy || 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default TradeHistory;
