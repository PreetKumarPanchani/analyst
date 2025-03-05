// components/charts/ForecastChart.tsx
import React from 'react';
import { 
  ComposedChart, 
  Line, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { formatCurrency, formatDate } from '@/lib/utils';
import { ForecastPoint } from '@/lib/api';

interface ForecastChartProps {
  historicalData: { date: string; sales: number }[];
  forecastData: ForecastPoint[];
  title?: string;
  height?: number;
  loading?: boolean;
}

const ForecastChart: React.FC<ForecastChartProps> = ({ 
  historicalData, 
  forecastData,
  title = 'Sales Forecast',
  height = 400,
  loading = false
}) => {
  // Combine historical and forecast data
  const combinedData = [
    ...historicalData.map(item => ({
      ...item,
      type: 'historical',
      salesForecast: null,
      salesLower: null,
      salesUpper: null
    })),
    ...forecastData.map(item => ({
      date: item.date,
      sales: null,
      salesForecast: item.sales,
      salesLower: item.sales_lower,
      salesUpper: item.sales_upper,
      type: 'forecast'
    }))
  ];
  
  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const dateStr = formatDate(label);
      const isHistorical = payload[0]?.payload.type === 'historical';
      
      return (
        <div className="bg-white p-3 border border-gray-200 shadow-md rounded-md">
          <p className="font-medium">{dateStr}</p>
          {isHistorical ? (
            // Historical data
            <p style={{ color: '#0088FE' }}>
              <span className="font-medium">Actual Sales: </span>
              {formatCurrency(payload[0]?.value || 0)}
            </p>
          ) : (
            // Forecast data
            <>
              <p style={{ color: '#8884d8' }}>
                <span className="font-medium">Forecast: </span>
                {formatCurrency(payload[0]?.value || 0)}
              </p>
              <p style={{ color: '#82ca9d' }}>
                <span className="font-medium">Range: </span>
                {formatCurrency(payload[1]?.payload.salesLower || 0)} - {formatCurrency(payload[2]?.payload.salesUpper || 0)}
              </p>
            </>
          )}
        </div>
      );
    }
    return null;
  };
  
  // Format x-axis ticks
  const formatXAxis = (value: string) => {
    const date = new Date(value);
    return `${date.getDate()}/${date.getMonth() + 1}`;
  };
  
  // Format y-axis ticks
  const formatYAxis = (value: number) => {
    return `£${value}`;
  };
  
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
        <ComposedChart
          data={combinedData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="date" 
            tickFormatter={formatXAxis}
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            tickFormatter={formatYAxis}
            tick={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {/* Historical data */}
          <Line 
            type="monotone" 
            dataKey="sales" 
            name="Historical Sales"
            stroke="#0088FE" 
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          
          {/* Forecast confidence interval */}
          <Area 
            type="monotone" 
            dataKey="salesUpper" 
            name="Forecast Range" 
            stroke="none"
            fill="#82ca9d" 
            fillOpacity={0.2}
          />
          <Area 
            type="monotone" 
            dataKey="salesLower" 
            stroke="none"
            fill="#82ca9d" 
            fillOpacity={0}
          />
          
          {/* Forecast line */}
          <Line 
            type="monotone" 
            dataKey="salesForecast" 
            name="Sales Forecast"
            stroke="#8884d8" 
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
            strokeDasharray="5 5"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ForecastChart;