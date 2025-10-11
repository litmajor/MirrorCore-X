
import React from 'react';
import { useAPI } from '../hooks/useAPI';
import { TrendingUp, TrendingDown, Clock, Target } from 'lucide-react';

const Trading: React.FC = () => {
  const { data: signals } = useAPI('/api/signals/active');

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Trading Signals</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {signals?.signals?.map((signal: any, index: number) => (
          <div key={index} className="bg-slate-800/50 backdrop-blur-xl border border-slate-700 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">{signal.symbol}</h3>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                signal.type === 'BUY' ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'
              }`}>
                {signal.type}
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-blue-400" />
                <span className="text-slate-400">Strength:</span>
                <span className="text-white font-medium">{(signal.strength * 100).toFixed(0)}%</span>
              </div>
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-purple-400" />
                <span className="text-slate-400">Confidence:</span>
                <span className="text-white font-medium">{(signal.confidence * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Trading;
