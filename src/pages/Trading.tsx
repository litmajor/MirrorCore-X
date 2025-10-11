
import React from 'react';
import { useAPI } from '../hooks/useAPI';
import { TrendingUp, TrendingDown, Clock, Target, AlertCircle } from 'lucide-react';
import { CardSkeleton } from '../components/Skeleton';

const Trading: React.FC = () => {
  const { data: signals, loading, error } = useAPI('/api/signals/active');

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">Trading Signals</h1>
        <p className="page-subtitle">Active market opportunities with technical analysis</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6">
        {loading ? (
          <>
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
          </>
        ) : error || !signals?.signals || signals.signals.length === 0 ? (
          <div className="col-span-full">
            <div className="brand-card text-center py-12">
              <AlertCircle className="w-16 h-16 text-txt-secondary mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">
                {error ? 'Unable to Load Signals' : 'No Active Signals'}
              </h3>
              <p className="text-txt-secondary">
                {error 
                  ? 'The trading backend is not running. Start the backend server to see live trading signals.' 
                  : 'No trading opportunities detected at the moment. The scanner is actively monitoring markets.'}
              </p>
            </div>
          </div>
        ) : (
          signals.signals.map((signal: any, index: number) => (
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

              <div className="border-t border-brand-cyan/20 pt-3 mt-3">
                <p className="text-xs text-txt-secondary mb-2">Signal Type: {signal.type || 'Scanner'}</p>
                <p className="text-xs text-txt-secondary">Timestamp: {new Date(signal.timestamp).toLocaleTimeString()}</p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Trading;
