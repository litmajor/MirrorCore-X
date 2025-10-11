
import React from 'react';
import { Settings, TrendingUp, Award, Target } from 'lucide-react';
import { useAPI } from '../hooks/useAPI';

const Optimization: React.FC = () => {
  const { data: optimizationData } = useAPI('/api/optimization/results');

  return (
    <div className="space-y-6">
      <div className="page-header">
        <h1 className="page-title">System Optimization</h1>
        <p className="page-subtitle">Comprehensive parameter optimization results</p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-txt-secondary text-sm">Total Parameters</h3>
            <Settings className="w-5 h-5 text-brand-cyan" />
          </div>
          <p className="text-3xl font-bold text-white">
            {optimizationData?.total_parameters || 0}
          </p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-txt-secondary text-sm">Optimized Components</h3>
            <Target className="w-5 h-5 text-brand-purple" />
          </div>
          <p className="text-3xl font-bold text-white">
            {optimizationData?.optimized_count || 0}
          </p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-txt-secondary text-sm">Avg Improvement</h3>
            <TrendingUp className="w-5 h-5 text-success" />
          </div>
          <p className="text-3xl font-bold text-success">
            {optimizationData?.avg_improvement ? `+${(optimizationData.avg_improvement * 100).toFixed(1)}%` : 'N/A'}
          </p>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-txt-secondary text-sm">Best Component</h3>
            <Award className="w-5 h-5 text-warning" />
          </div>
          <p className="text-lg font-bold text-white">
            {optimizationData?.best_component?.name || 'N/A'}
          </p>
        </div>
      </div>

      {/* Component Results */}
      <div className="brand-card">
        <h2 className="text-xl font-semibold text-white mb-6">Component Optimization Results</h2>
        
        <div className="space-y-4">
          {optimizationData?.components?.map((component: any, idx: number) => (
            <div key={idx} className="p-4 bg-bg-surface rounded-lg border border-accent-dark">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">{component.name}</h3>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  component.improvement > 0 ? 'bg-success/20 text-success' : 
                  component.improvement < 0 ? 'bg-error/20 text-error' : 
                  'bg-neutral/20 text-neutral'
                }`}>
                  {component.improvement > 0 ? '+' : ''}{(component.improvement * 100).toFixed(1)}%
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-xs text-txt-secondary mb-1">Parameters Optimized</p>
                  <p className="text-white font-semibold">{component.params_optimized}</p>
                </div>
                <div>
                  <p className="text-xs text-txt-secondary mb-1">Iterations</p>
                  <p className="text-white font-semibold">{component.iterations}</p>
                </div>
              </div>

              {component.parameters && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-txt-secondary mb-2">Optimized Parameters:</p>
                  {Object.entries(component.parameters).map(([param, values]: [string, any]) => (
                    <div key={param} className="flex justify-between text-xs">
                      <span className="text-txt-secondary">{param}</span>
                      <span className="text-white">
                        {values.old} → <span className="text-brand-cyan">{values.new}</span>
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="brand-card">
          <h2 className="text-xl font-semibold text-white mb-4">Optimization by Category</h2>
          <div className="space-y-3">
            {optimizationData?.categories?.map((cat: any, idx: number) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-bg-surface rounded-lg">
                <span className="text-white font-medium">{cat.name}</span>
                <div className="text-right">
                  <p className="text-white font-semibold">{cat.count} params</p>
                  <p className={`text-xs ${cat.improvement > 0 ? 'text-success' : 'text-error'}`}>
                    {cat.improvement > 0 ? '+' : ''}{(cat.improvement * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="brand-card">
          <h2 className="text-xl font-semibold text-white mb-4">Top Improvements</h2>
          <div className="space-y-3">
            {optimizationData?.top_improvements?.map((imp: any, idx: number) => (
              <div key={idx} className="p-3 bg-bg-surface rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-white font-medium">{imp.component}</span>
                  <span className="text-success font-semibold">+{(imp.improvement * 100).toFixed(1)}%</span>
                </div>
                <p className="text-xs text-txt-secondary">{imp.parameter}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Optimization;
