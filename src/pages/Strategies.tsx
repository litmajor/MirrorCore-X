
import React, { useState } from 'react';
import { Layers, Play, Pause, Settings, TrendingUp, TrendingDown } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';

const Strategies: React.FC = () => {
  const { data: strategies } = useAPI('/api/strategies');
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-600/20 text-green-400';
      case 'paused': return 'bg-yellow-600/20 text-yellow-400';
      default: return 'bg-slate-600/20 text-slate-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Trading Strategies</h1>
        <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white font-medium transition-colors">
          Add Strategy
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {strategies?.strategies?.map((strategy: any) => (
          <div 
            key={strategy.name}
            className="bg-slate-800/50 backdrop-blur-xl border border-slate-700 rounded-xl p-6 hover:border-blue-500/50 transition-all cursor-pointer"
            onClick={() => setSelectedStrategy(strategy.name)}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <Layers className="w-6 h-6 text-blue-400" />
                <h3 className="text-xl font-semibold text-white">{strategy.name}</h3>
              </div>
              <div className="flex items-center space-x-3">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(strategy.status)}`}>
                  {strategy.status}
                </span>
                <button className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
                  <Settings className="w-5 h-5 text-slate-400" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-slate-400 mb-1">P&L</p>
                <p className={`font-semibold ${strategy.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${strategy.pnl?.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-slate-400 mb-1">Win Rate</p>
                <p className="font-semibold text-white">{strategy.win_rate?.toFixed(1)}%</p>
              </div>
              <div>
                <p className="text-slate-400 mb-1">Status</p>
                <div className="flex items-center space-x-2">
                  {strategy.status === 'active' ? (
                    <Play className="w-4 h-4 text-green-400" />
                  ) : (
                    <Pause className="w-4 h-4 text-yellow-400" />
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
