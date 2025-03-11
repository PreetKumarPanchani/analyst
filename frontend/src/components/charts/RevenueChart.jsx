import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { formatCurrency } from '../../utils/formatters';
import { processForecastResponse } from '@/utils/dataProcessing';

/**
 * Revenue chart component for displaying revenue data over time
 * 
 * @param {Object} props
 * @param {Array} props.data - Revenue data array with date, actual, and predicted values
 * @param {number} props.height - Chart height (default: 400)
 * @param {boolean} props.showLegend - Whether to show the legend (default: true)
 * @param {boolean} props.showTooltip - Whether to show tooltips (default: true)
 * @param {boolean} props.showGrid - Whether to show the grid (default: true)
 */
const RevenueChart = ({ 
  data = [], 
  height = 400, 
  showLegend = true, 
  showTooltip = true, 
  showGrid = true 
}) => {
  
  // Find the forecast start index (where actuals end)
  const forecastStartIndex = data.findIndex(item => item.actual === null && item.predicted !== null);
  
  const formatXAxis = (dateStr) => {
    const date = new Date(dateStr);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };
  
  const formatTooltip = (value) => {
    if (value === null || value === undefined) return ['-', ''];
    return [formatCurrency(value), ''];
  };
  
  const formatTooltipLabel = (label) => {
    const date = new Date(label);
    return date.toLocaleDateString();
  };
  
  const prepareChartData = (data) => {
    if (!data || !data.dates) return [];
    
    // Process data to ensure it's safe
    const processedData = processForecastResponse(data);
    
    // Now create your chart data
    return processedData.dates.map((date, index) => ({
      date,
      actual: processedData.actuals[index] || null,
      prediction: processedData.predictions[index] || null,
      // ...other properties
    }));
  };
  
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          {showGrid && <CartesianGrid strokeDasharray="3 3" />}
          <XAxis 
            dataKey="date" 
            tickFormatter={formatXAxis} 
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            tickFormatter={(value) => formatCurrency(value, { decimals: 0, showSymbol: false })}
            tick={{ fontSize: 12 }}
          />
          {showTooltip && (
            <Tooltip 
              formatter={formatTooltip}
              labelFormatter={formatTooltipLabel}
            />
          )}
          {showLegend && <Legend />}
          
          <Line 
            type="monotone" 
            dataKey="actual" 
            name="Actual Revenue" 
            stroke="#3b82f6" 
            strokeWidth={2} 
            dot={{ r: 3 }}
            activeDot={{ r: 6 }}
          />
          <Line 
            type="monotone" 
            dataKey="predicted" 
            name="Forecast" 
            stroke="#10b981" 
            strokeWidth={2}
            dot={(props) => {
              // Only show dots for forecast points
              const index = props.index;
              if (forecastStartIndex !== -1 && index >= forecastStartIndex) {
                return <circle {...props} r={3} />;
              }
              return null;
            }}
            strokeDasharray={(props) => {
              // Make the line dashed for forecast part
              const index = props?.index;
              if (forecastStartIndex !== -1 && index >= forecastStartIndex) {
                return "5 5";
              }
              return "0";
            }}
          />
          {data[0]?.upperBound && (
            <Line 
              type="monotone" 
              dataKey="upperBound" 
              name="Upper Bound" 
              stroke="#d1d5db" 
              strokeWidth={1}
              dot={false}
              activeDot={false}
            />
          )}
          {data[0]?.lowerBound && (
            <Line 
              type="monotone" 
              dataKey="lowerBound" 
              name="Lower Bound" 
              stroke="#d1d5db" 
              strokeWidth={1}
              dot={false}
              activeDot={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RevenueChart;