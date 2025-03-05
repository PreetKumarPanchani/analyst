// components/charts/CategoryChart.tsx
import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { formatCurrency, getChartColor } from '@/lib/utils';

interface CategoryData {
  category: string;
  sales: number;
  quantity: number;
  product_count: number;
}

interface CategoryChartProps {
  data: CategoryData[];
  dataKey?: 'sales' | 'quantity' | 'product_count';
  title?: string;
  height?: number;
  loading?: boolean;
}

const CategoryChart: React.FC<CategoryChartProps> = ({ 
  data, 
  dataKey = 'sales',
  title = 'Sales by Category',
  height = 300,
  loading = false
}) => {
  // Sort data by the selected data key
  const sortedData = [...data]
    .sort((a, b) => b[dataKey] - a[dataKey])
    .slice(0, 10);  // Only show top 10 categories
  
  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 shadow-md rounded-md">
          <p className="font-medium">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={`item-${index}`} style={{ color: entry.color }}>
              <span className="font-medium">{entry.name}: </span>
              {entry.dataKey === 'sales' 
                ? formatCurrency(entry.value) 
                : entry.value.toLocaleString()}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };
  
  // Format y-axis ticks
  const formatYAxis = (value: number) => {
    if (dataKey === 'sales') {
      return `£${value}`;
    }
    return value;
  };
  
  // Get name for the data key
  const dataKeyName = {
    'sales': 'Sales',
    'quantity': 'Quantity',
    'product_count': 'Product Count'
  }[dataKey];
  
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
        <BarChart
          data={sortedData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis type="number" tickFormatter={(value: number) => formatYAxis(value).toString()} />
          <YAxis 
            dataKey="category" 
            type="category" 
            tick={{ fontSize: 12 }}
            width={80}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Bar 
            dataKey={dataKey} 
            name={dataKeyName}
            fill={getChartColor(0)} 
            radius={[0, 4, 4, 0]} 
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CategoryChart;