import React from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * Comparison chart component for comparing multiple data series
 * 
 * @param {Object} props
 * @param {Array} props.data - Chart data array
 * @param {Array} props.series - Array of series configurations [{ dataKey, name, color, type }]
 * @param {string} props.xAxisKey - Key to use for the x-axis
 * @param {string} props.chartType - Chart type ('bar' or 'line')
 * @param {number} props.height - Chart height (default: 400)
 * @param {boolean} props.stacked - Whether to stack bars (only for bar charts)
 * @param {Object} props.formatters - Custom formatters for axis and tooltips
 */
const ComparisonChart = ({ 
  data = [], 
  series = [],
  xAxisKey = 'name',
  chartType = 'bar',
  height = 400,
  stacked = false,
  formatters = {
    xAxis: (value) => value,
    yAxis: (value) => value,
    tooltip: (value) => [value, '']
  }
}) => {
  // Default colors if not provided in series
  const defaultColors = [
    '#3b82f6', // blue
    '#10b981', // green
    '#8b5cf6', // purple
    '#f59e0b', // amber
    '#ef4444', // red
    '#6b7280', // gray
  ];
  
  // Ensure all series have colors
  const enrichedSeries = series.map((item, index) => ({
    ...item,
    color: item.color || defaultColors[index % defaultColors.length]
  }));
  
  // Common props for all chart types
  const commonProps = {
    data,
    margin: { top: 10, right: 30, left: 0, bottom: 0 },
  };
  
  // Render chart based on chartType
  const renderChart = () => {
    if (chartType === 'bar') {
      return (
        <BarChart {...commonProps}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey={xAxisKey} 
            tickFormatter={formatters.xAxis}
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            tickFormatter={formatters.yAxis}
            tick={{ fontSize: 12 }}
          />
          <Tooltip formatter={formatters.tooltip} />
          <Legend />
          {enrichedSeries.map((item, index) => (
            <Bar 
              key={index}
              dataKey={item.dataKey}
              name={item.name || item.dataKey}
              fill={item.color}
              stackId={stacked ? 'stack' : undefined}
            />
          ))}
        </BarChart>
      );
    }
    
    // Default to line chart
    return (
      <LineChart {...commonProps}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey={xAxisKey} 
          tickFormatter={formatters.xAxis}
          tick={{ fontSize: 12 }}
        />
        <YAxis 
          tickFormatter={formatters.yAxis}
          tick={{ fontSize: 12 }}
        />
        <Tooltip formatter={formatters.tooltip} />
        <Legend />
        {enrichedSeries.map((item, index) => (
          <Line 
            key={index}
            type="monotone"
            dataKey={item.dataKey}
            name={item.name || item.dataKey}
            stroke={item.color}
            strokeWidth={2}
            dot={(props) => <circle {...props} r={3} />}
            activeDot={(props) => <circle {...props} r={6} />}
            
            
          />
        ))}
      </LineChart>
    );
  };
  
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
};

export default ComparisonChart;