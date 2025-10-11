import React, { useState } from 'react';
import { toast } from 'sonner';
import { Sparkles, CheckCircle, XCircle, Info } from 'lucide-react';
import { MetricCardSkeleton, ChartSkeleton, CardSkeleton, TableRowSkeleton } from '../components/Skeleton';
import { useTheme } from '../contexts/ThemeContext';

const UXDemo: React.FC = () => {
  const [showSkeletons, setShowSkeletons] = useState(false);
  const { theme } = useTheme();

  const showSuccessToast = () => {
    toast.success('Trade Executed Successfully', {
      description: 'BUY BTC @ $45,000 | P&L: +$1,250.00',
      duration: 5000,
    });
  };

  const showErrorToast = () => {
    toast.error('Trade Failed', {
      description: 'Insufficient balance for this order',
      duration: 5000,
    });
  };

  const showInfoToast = () => {
    toast.info('Market Alert', {
      description: 'New signal detected for ETH/USDT',
      duration: 5000,
    });
  };

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">UX Features Demo</h1>
        <p className="page-subtitle">Explore all the new user experience enhancements</p>
      </div>

      {/* Theme Display */}
      <div className="brand-card">
        <h2 className="text-xl font-semibold text-white mb-4">Current Theme</h2>
        <div className="flex items-center space-x-4">
          <div className="px-6 py-3 bg-brand-cyan/20 rounded-lg border border-brand-cyan/40">
            <p className="text-brand-cyan font-semibold">
              {theme === 'dark' ? '🌙 Dark Mode' : '☀️ Light Mode'}
            </p>
          </div>
          <p className="text-txt-secondary">
            Use the theme toggle in the top-right corner to switch modes
          </p>
        </div>
      </div>

      {/* Toast Notifications */}
      <div className="brand-card">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Sparkles className="w-6 h-6 mr-2 text-brand-cyan" />
          Toast Notifications
        </h2>
        <p className="text-txt-secondary mb-4">
          Click the buttons below to see different types of toast notifications:
        </p>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={showSuccessToast}
            className="btn-primary flex items-center space-x-2"
          >
            <CheckCircle className="w-4 h-4" />
            <span>Success Toast</span>
          </button>
          <button
            onClick={showErrorToast}
            className="px-6 py-3 bg-error text-white rounded-lg hover:bg-error/80 transition-all flex items-center space-x-2"
          >
            <XCircle className="w-4 h-4" />
            <span>Error Toast</span>
          </button>
          <button
            onClick={showInfoToast}
            className="btn-secondary flex items-center space-x-2"
          >
            <Info className="w-4 h-4" />
            <span>Info Toast</span>
          </button>
        </div>
      </div>

      {/* Loading Skeletons */}
      <div className="brand-card">
        <h2 className="text-xl font-semibold text-white mb-4">Loading Skeletons</h2>
        <p className="text-txt-secondary mb-4">
          Toggle to see loading skeleton animations:
        </p>
        <button
          onClick={() => setShowSkeletons(!showSkeletons)}
          className="btn-primary mb-6"
        >
          {showSkeletons ? 'Hide' : 'Show'} Skeletons
        </button>

        <div className="space-y-6">
          {/* Metric Card Skeletons */}
          <div>
            <h3 className="text-sm font-semibold text-txt-secondary mb-3">Metric Cards</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {showSkeletons ? (
                <>
                  <MetricCardSkeleton />
                  <MetricCardSkeleton />
                  <MetricCardSkeleton />
                  <MetricCardSkeleton />
                </>
              ) : (
                <>
                  <div className="metric-card">
                    <p className="text-txt-secondary text-sm mb-2">Total P&L</p>
                    <p className="text-2xl font-bold text-success">$12,450.00</p>
                  </div>
                  <div className="metric-card">
                    <p className="text-txt-secondary text-sm mb-2">Win Rate</p>
                    <p className="text-2xl font-bold text-white">68.5%</p>
                  </div>
                  <div className="metric-card">
                    <p className="text-txt-secondary text-sm mb-2">Active Signals</p>
                    <p className="text-2xl font-bold text-brand-cyan">12</p>
                  </div>
                  <div className="metric-card">
                    <p className="text-txt-secondary text-sm mb-2">Sharpe Ratio</p>
                    <p className="text-2xl font-bold text-white">2.34</p>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Chart Skeleton */}
          <div>
            <h3 className="text-sm font-semibold text-txt-secondary mb-3">Chart Skeleton</h3>
            {showSkeletons ? (
              <ChartSkeleton />
            ) : (
              <div className="chart-container">
                <h4 className="text-lg font-semibold text-white mb-2">Performance Chart</h4>
                <div className="h-64 flex items-center justify-center bg-bg-surface/30 rounded-lg">
                  <p className="text-txt-secondary">Chart would render here</p>
                </div>
              </div>
            )}
          </div>

          {/* Card Skeleton */}
          <div>
            <h3 className="text-sm font-semibold text-txt-secondary mb-3">Generic Card</h3>
            {showSkeletons ? (
              <CardSkeleton />
            ) : (
              <div className="brand-card">
                <h4 className="text-lg font-semibold text-white mb-2">Trade Signal</h4>
                <p className="text-txt-secondary">BUY BTC/USDT at $45,000</p>
              </div>
            )}
          </div>

          {/* Table Row Skeleton */}
          <div>
            <h3 className="text-sm font-semibold text-txt-secondary mb-3">Table Rows</h3>
            <div className="glass rounded-lg overflow-hidden">
              <table className="w-full">
                <thead className="bg-bg-surface/50">
                  <tr>
                    <th className="px-4 py-3 text-left text-txt-secondary text-sm">Symbol</th>
                    <th className="px-4 py-3 text-left text-txt-secondary text-sm">Side</th>
                    <th className="px-4 py-3 text-left text-txt-secondary text-sm">Price</th>
                    <th className="px-4 py-3 text-left text-txt-secondary text-sm">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {showSkeletons ? (
                    <>
                      <TableRowSkeleton />
                      <TableRowSkeleton />
                      <TableRowSkeleton />
                    </>
                  ) : (
                    <>
                      <tr className="border-b border-bg-surface">
                        <td className="px-4 py-3 text-white">BTC/USDT</td>
                        <td className="px-4 py-3 text-success">BUY</td>
                        <td className="px-4 py-3 text-white">$45,000</td>
                        <td className="px-4 py-3 text-success">+$1,250</td>
                      </tr>
                      <tr className="border-b border-bg-surface">
                        <td className="px-4 py-3 text-white">ETH/USDT</td>
                        <td className="px-4 py-3 text-error">SELL</td>
                        <td className="px-4 py-3 text-white">$2,800</td>
                        <td className="px-4 py-3 text-error">-$340</td>
                      </tr>
                    </>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Responsive Info */}
      <div className="brand-card">
        <h2 className="text-xl font-semibold text-white mb-4">Mobile Responsive</h2>
        <p className="text-txt-secondary mb-4">
          The interface is fully optimized for mobile devices:
        </p>
        <ul className="space-y-2 text-txt-secondary">
          <li className="flex items-start">
            <CheckCircle className="w-5 h-5 text-success mr-2 mt-0.5 flex-shrink-0" />
            <span>Sidebar collapses on mobile with overlay navigation</span>
          </li>
          <li className="flex items-start">
            <CheckCircle className="w-5 h-5 text-success mr-2 mt-0.5 flex-shrink-0" />
            <span>Responsive grids (1 column on mobile, 2-4 on larger screens)</span>
          </li>
          <li className="flex items-start">
            <CheckCircle className="w-5 h-5 text-success mr-2 mt-0.5 flex-shrink-0" />
            <span>Touch-friendly buttons and interactions</span>
          </li>
          <li className="flex items-start">
            <CheckCircle className="w-5 h-5 text-success mr-2 mt-0.5 flex-shrink-0" />
            <span>Optimized spacing and padding for smaller screens</span>
          </li>
          <li className="flex items-start">
            <CheckCircle className="w-5 h-5 text-success mr-2 mt-0.5 flex-shrink-0" />
            <span>Responsive charts and visualizations</span>
          </li>
        </ul>
        <p className="text-txt-secondary mt-4 text-sm italic">
          Try resizing your browser window or viewing on a mobile device to see the responsive design in action.
        </p>
      </div>
    </div>
  );
};

export default UXDemo;
