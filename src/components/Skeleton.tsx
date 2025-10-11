import React from 'react';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'rectangular',
  width,
  height,
  animation = 'pulse'
}) => {
  const baseClasses = 'bg-bg-surface/50';
  
  const variantClasses = {
    text: 'rounded h-4',
    circular: 'rounded-full',
    rectangular: 'rounded-lg'
  };

  const animationClasses = {
    pulse: 'animate-pulse',
    wave: 'animate-shimmer',
    none: ''
  };

  const style = {
    width: typeof width === 'number' ? `${width}px` : width,
    height: typeof height === 'number' ? `${height}px` : height
  };

  return (
    <div
      className={`${baseClasses} ${variantClasses[variant]} ${animationClasses[animation]} ${className}`}
      style={style}
    />
  );
};

export const MetricCardSkeleton: React.FC = () => (
  <div className="metric-card">
    <div className="flex items-center justify-between mb-4">
      <Skeleton width={100} height={16} />
      <Skeleton variant="circular" width={20} height={20} />
    </div>
    <div className="flex items-end justify-between">
      <div className="w-full">
        <Skeleton width={120} height={36} className="mb-2" />
        <Skeleton width={80} height={20} />
      </div>
    </div>
  </div>
);

export const ChartSkeleton: React.FC = () => (
  <div className="chart-container">
    <Skeleton width={200} height={24} className="mb-4" />
    <Skeleton width="100%" height={300} />
  </div>
);

export const TableRowSkeleton: React.FC = () => (
  <tr className="border-b border-bg-surface">
    <td className="px-4 py-3"><Skeleton width={100} /></td>
    <td className="px-4 py-3"><Skeleton width={80} /></td>
    <td className="px-4 py-3"><Skeleton width={120} /></td>
    <td className="px-4 py-3"><Skeleton width={90} /></td>
  </tr>
);

export const CardSkeleton: React.FC = () => (
  <div className="brand-card">
    <Skeleton width="60%" height={24} className="mb-3" />
    <Skeleton width="100%" height={16} className="mb-2" />
    <Skeleton width="80%" height={16} />
  </div>
);
