
import React, { useState } from 'react';
import { Layers, Play, Pause, Settings, TrendingUp, TrendingDown } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';

const Strategies: React.FC = () => {
  const { data: strategies } = useAPI('/api/strategies');
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-success/20 text-success';
      case 'paused': return 'bg-warning/20 text-warning';
      default: return 'bg-neutral/20 text-neutral';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between page-header">
        <div>
          <h1 className="page-title">Trading Strategies</h1>
          <p className="page-subtitle">Manage and monitor active trading strategies</p>
        </div>
        <button className="btn-primary">
          Add Strategy
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {strategies?.strategies?.map((strategy: any) => (
          <div 
            key={strategy.name}
            className="brand-card cursor-pointer"
            onClick={() => setSelectedStrategy(strategy.name)}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <Layers className="w-6 h-6 text-brand-cyan" />
                <h3 className="text-xl font-semibold text-white">{strategy.name}</h3>
              </div>
              <div className="flex items-center space-x-3">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(strategy.status)}`}>
                  {strategy.status}
                </span>
                <button className="p-2 hover:bg-bg-surface rounded-lg transition-colors">
                  <Settings className="w-5 h-5 text-txt-secondary" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-txt-secondary mb-1">P&L</p>
                <p className={`font-semibold ${strategy.pnl >= 0 ? 'text-success' : 'text-error'}`}>
                  ${strategy.pnl?.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-txt-secondary mb-1">Win Rate</p>
                <p className="font-semibold text-white">{strategy.win_rate?.toFixed(1)}%</p>
              </div>
              <div>
                <p className="text-txt-secondary mb-1">Status</p>
                <div className="flex items-center space-x-2">
                  {strategy.status === 'active' ? (
                    <Play className="w-4 h-4 text-success" />
                  ) : (
                    <Pause className="w-4 h-4 text-warning" />
                  )}
                  <span className="text-white">{strategy.status}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Strategies;
