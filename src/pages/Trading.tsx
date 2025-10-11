
import React from 'react';
import { useAPI } from '../hooks/useAPI';
import { TrendingUp, TrendingDown, Clock, Target } from 'lucide-react';

const Trading: React.FC = () => {
  const { data: signals } = useAPI('/api/signals/active');

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Trading Signals</h1>
        <p className="page-subtitle">Active market opportunities with technical analysis</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {signals?.signals?.map((signal: any, index: number) => (
          <div key={index} className={signal.signal === 'BUY' || signal.signal?.includes('Buy') ? 'signal-buy' : 'signal-sell'}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">{signal.symbol}</h3>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                signal.signal === 'BUY' || signal.signal?.includes('Buy') ? 'bg-success/20 text-success' : 'bg-error/20 text-error'
              }`}>
                {signal.signal || signal.type}
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm mb-4">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-brand-cyan" />
                <span className="text-txt-secondary">Strength:</span>
                <span className="text-white font-medium">{((signal.strength || 0) * 100).toFixed(0)}%</span>
              </div>
              <div className="flex items-center space-x-2">
                <Clock className="w-4 h-4 text-brand-purple" />
                <span className="text-txt-secondary">Price:</span>
                <span className="text-white font-medium">${signal.price?.toFixed(2)}</span>
              </div>
            </div>

            {/* Additional Technical Details */}
            <div className="border-t border-brand-cyan/20 pt-3 mt-3">
              <p className="text-xs text-txt-secondary mb-2">Signal Type: {signal.type || 'Scanner'}</p>
              <p className="text-xs text-txt-secondary">Timestamp: {new Date(signal.timestamp).toLocaleTimeString()}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Trading;
