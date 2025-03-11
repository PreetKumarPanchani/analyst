import React, { useEffect, useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Brush } from 'recharts';
import { processForecastResponse } from '@/utils/dataProcessing';
import DebugObject from '@/components/DebugObject';

/**
 * Forecast chart component for displaying forecast data with various options
 * 
 * @param {Object} props
 * @param {Array} props.data - Chart data array
 * @param {string} props.type - Chart type ('line' or 'bar')
 * @param {string} props.dataKey - Key to use for the main data series
 * @param {string} props.nameKey - Key to use for item names (x-axis)
 * @param {Object} props.colors - Custom colors for chart elements
 * @param {number} props.height - Chart height (default: 400)
 * @param {boolean} props.showBounds - Whether to show upper/lower bounds (default: true)
 * @param {boolean} props.showBrush - Whether to show the brush for zooming (default: false)
 * @param {Object} props.formatter - Custom formatters for tooltip values
 */
const ForecastChart = ({ 
  data = [], 
  type = 'line', 
  dataKey = 'value',
  nameKey = 'date',
  colors = {
    main: '#10b981', // green
    actual: '#3b82f6', // blue
    bounds: '#d1d5db', // light gray
  },
  height = 400,
  showBounds = true,
  showBrush = false,
  formatter = {
    xAxis: (value) => {
      if (typeof value === 'string' && value.includes('-')) {
        // Likely a date
        const date = new Date(value);
        return `${date.getMonth() + 1}/${date.getDate()}`;
      }
      return value;
    },
    tooltip: (value) => [value !== null ? value.toFixed(2) : '-', ''],
    tooltipLabel: (label) => {
      if (typeof label === 'string' && label.includes('-')) {
        // Likely a date
        return new Date(label).toLocaleDateString();
      }
      return label;
    }
  }
}) => {
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    if (data?.components) {
      // Process the data to ensure it's safe for rendering
      const processedData = processForecastResponse(data);
      setChartData(processedData);
    }
  }, [data]);

  // Determine if the chart should show forecast indicators
  const hasForecast = data.some(item => item.isForecast || item.actual === null);
  
  // Find where forecast starts
  const forecastStartIndex = data.findIndex(item => 
    (item.isForecast === true) || 
    (item.actual === null && item.predicted !== null)
  );
  
  // Common props for all chart types
  const commonProps = {
    data,
    margin: { top: 10, right: 30, left: 0, bottom: 0 },
  };
  
  // Render chart based on type
  const renderChart = () => {
    if (type === 'bar') {
      return (
        <BarChart {...commonProps}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey={nameKey} 
            tickFormatter={formatter.xAxis}
            tick={{ fontSize: 12 }}
          />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip 
            formatter={formatter.tooltip}
            labelFormatter={formatter.tooltipLabel}
          />
          <Legend />
          {data[0]?.actual !== undefined && (
            <Bar 
              dataKey="actual" 
              name="Actual" 
              fill={colors.actual} 
            />
          )}
          <Bar 
            dataKey={dataKey || 'predicted'} 
            name="Forecast" 
            fill={colors.main} 
          />
          {showBrush && <Brush dataKey={nameKey} height={30} stroke="#8884d8" />}
        </BarChart>
      );
    }
    
    // Default to line chart
    return (
      <LineChart {...commonProps}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey={nameKey} 
          tickFormatter={formatter.xAxis}
          tick={{ fontSize: 12 }}
        />
        <YAxis tick={{ fontSize: 12 }} />
        <Tooltip 
          formatter={formatter.tooltip}
          labelFormatter={formatter.tooltipLabel}
        />
        <Legend />
        {data[0]?.actual !== undefined && (
          <Line 
            type="monotone" 
            dataKey="actual" 
            name="Actual" 
            stroke={colors.actual} 
            strokeWidth={2} 
            dot={{ r: 3 }}
            activeDot={{ r: 6 }}
          />
        )}
        <Line 
          type="monotone" 
          dataKey={dataKey || 'predicted'} 
          name="Forecast" 
          stroke={colors.main} 
          strokeWidth={2}
          strokeDasharray={(props) => {
            // Make the line dashed for forecast part
            const index = props?.index;
            if (hasForecast && forecastStartIndex !== -1 && index >= forecastStartIndex) {
              return "5 5";
            }
            return "0";
          }}
          dot={(props) => {
            // Only show dots for forecast points if we have a mixed chart
            if (!hasForecast) return { r: 3 };
            
            const index = props.index;
            if (forecastStartIndex !== -1 && index >= forecastStartIndex) {
              return <circle {...props} r={3} />;
            }
            return null;
          }}
        />
        {showBounds && data[0]?.upperBound && (
          <Line 
            type="monotone" 
            dataKey="upperBound" 
            name="Upper Bound" 
            stroke={colors.bounds} 
            strokeWidth={1}
            dot={false}
          />
        )}
        {showBounds && data[0]?.lowerBound && (
          <Line 
            type="monotone" 
            dataKey="lowerBound" 
            name="Lower Bound" 
            stroke={colors.bounds} 
            strokeWidth={1}
            dot={false}
          />
        )}
        {showBrush && <Brush dataKey={nameKey} height={30} stroke="#8884d8" />}
      </LineChart>
    );
  };
  
  // Safe rendering in case of issues
  if (!chartData) return <div>Loading chart data...</div>;
  if (typeof chartData === 'object' && chartData.error) {
    return <div className="error-message">{chartData.error}</div>;
  }

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
};

export default ForecastChart;