// components/ui/StatsCard.tsx
import React from 'react';

interface StatsCardProps {
  title: string;
  value: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    label: string;
  };
  className?: string;
}

const StatsCard: React.FC<StatsCardProps> = ({ 
  title, 
  value, 
  icon, 
  trend, 
  className = '' 
}) => {
  const trendColor = trend && trend.value >= 0 ? 'text-green-500' : 'text-red-500';
  const trendSymbol = trend && trend.value >= 0 ? '↑' : '↓';

  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-sm font-medium text-gray-400">{title}</h3>
          <p className="text-2xl font-bold mt-1">{value}</p>
        </div>
        {icon && (
          <div className="p-2 bg-blue-50 rounded-md">
            {icon}
          </div>
        )}
      </div>
      
      {trend && (
        <div className="mt-4">
          <span className={`inline-flex items-center ${trendColor}`}>
            <span className="mr-1">{trendSymbol}</span>
            <span className="font-medium">{Math.abs(trend.value).toFixed(1)}%</span>
          </span>
          <span className="text-gray-500 text-sm ml-2">{trend.label}</span>
        </div>
      )}
    </div>
  );
};

export default StatsCard;