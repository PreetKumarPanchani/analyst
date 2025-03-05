// components/charts/SalesChart.tsx
import React, { useMemo } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { formatCurrency, formatMonth, formatDate } from '@/lib/utils';

interface DataPoint {
  date?: string;
  month?: string;
  sales: number;
  transactions?: number;
  avg_basket?: number;
}

interface SalesChartProps {
  data: DataPoint[];
  timeFrame: 'daily' | 'monthly';
  dataKey?: 'sales' | 'transactions' | 'avg_basket';
  title?: string;
  height?: number;
  loading?: boolean;
  showLegend?: boolean;
}

const SalesChart: React.FC<SalesChartProps> = ({ 
  data, 
  timeFrame, 
  dataKey = 'sales',
  title = 'Sales Over Time',
  height = 300,
  loading = false,
  showLegend = true
}) => {
  // Format x-axis ticks based on timeframe
  const formatXAxis = (value: string) => {
    if (timeFrame === 'monthly') {
      return formatMonth(value);
    } else {
      // Format to shorter date for daily view
      const date = new Date(value);
      return `${date.getDate()}/${date.getMonth() + 1}`;
    }
  };
  
  // Customize tooltip to show formatted values
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const dateLabel = timeFrame === 'monthly' 
        ? formatMonth(label)
        : formatDate(label);
      
      return (
        <div className="bg-white p-3 border border-gray-200 shadow-md rounded-md">
          <p className="font-medium">{dateLabel}</p>
          {payload.map((entry: any, index: number) => (
            <p key={`item-${index}`} style={{ color: entry.color }}>
              <span className="font-medium">{entry.name}: </span>
              {entry.dataKey === 'sales' 
                ? formatCurrency(entry.value) 
                : entry.dataKey === 'avg_basket'
                ? formatCurrency(entry.value)
                : entry.value.toLocaleString()}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };
  
  // Get the right data key name based on the timeframe
  const xAxisDataKey = timeFrame === 'monthly' ? 'month' : 'date';
  
  // Format y-axis ticks
  const formatYAxis = (value: number) => {
    if (dataKey === 'sales' || dataKey === 'avg_basket') {
      return `£${value}`;
    }
    return value;
  };
  
  // Get color based on data key
  const lineColor = useMemo(() => {
    switch (dataKey) {
      case 'sales': return '#0088FE';
      case 'transactions': return '#00C49F';
      case 'avg_basket': return '#FFBB28';
      default: return '#0088FE';
    }
  }, [dataKey]);
  
  // Get name for the data key
  const dataKeyName = useMemo(() => {
    switch (dataKey) {
      case 'sales': return 'Sales';
      case 'transactions': return 'Transactions';
      case 'avg_basket': return 'Avg. Basket';
      default: return 'Sales';
    }
  }, [dataKey]);
  
  if (loading) {
    return (
      <div 
        className="w-full bg-white rounded-lg shadow-md p-4 flex items-center justify-center"
        style={{ height: `${height}px` }}
      >
        <div className="text-gray-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg shadow-md p-4">
      <h3 className="text-lg font-medium text-gray-700 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey={xAxisDataKey} 
            tickFormatter={formatXAxis}
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            tickFormatter={(value: number) => formatYAxis(value).toString()}
            tick={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          {showLegend && <Legend />}
          <Line 
            type="monotone" 
            dataKey={dataKey} 
            name={dataKeyName}
            stroke={lineColor} 
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SalesChart;