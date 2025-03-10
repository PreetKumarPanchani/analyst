/**
 * Utility functions for chart data preparation and configuration
 */

import { formatDate, formatCurrency } from './formatters';

/**
 * Prepare data for a time series chart with actual and predicted values
 * 
 * @param {Object} forecastData - Forecast data from API
 * @param {Object} options - Additional options
 * @param {boolean} options.onlyForecast - Whether to only include future forecast data
 * @param {boolean} options.includeError - Whether to include upper and lower bounds
 * @returns {Array} Formatted chart data array
 */
export const prepareTimeSeriesData = (forecastData, options = {}) => {
  const { onlyForecast = false, includeError = true } = options;
  
  if (!forecastData || !forecastData.dates || !forecastData.predictions) {
    return [];
  }
  
  // Find the index where actuals end and forecasts begin
  const forecastStartIndex = forecastData.actuals.findLastIndex(val => val !== null) + 1;
  
  // Filter data based on onlyForecast option
  const startIndex = onlyForecast ? forecastStartIndex : 0;
  const dates = forecastData.dates.slice(startIndex);
  const actuals = forecastData.actuals.slice(startIndex);
  const predictions = forecastData.predictions.slice(startIndex);
  
  // Prepare chart data
  const chartData = dates.map((date, index) => {
    const dataPoint = {
      date,
      actual: actuals[index] || null,
      predicted: predictions[index],
      isForecast: index + startIndex >= forecastStartIndex
    };
    
    // Add upper and lower bounds if requested
    if (includeError && forecastData.upper_bound && forecastData.lower_bound) {
      dataPoint.upperBound = forecastData.upper_bound[index + startIndex];
      dataPoint.lowerBound = forecastData.lower_bound[index + startIndex];
    }
    
    return dataPoint;
  });
  
  return chartData;
};

/**
 * Prepare data for a bar chart
 * 
 * @param {Array} items - Array of items with names and values
 * @param {Object} options - Additional options
 * @param {string} options.nameKey - Key for item name
 * @param {string} options.valueKey - Key for item value
 * @param {number} options.limit - Maximum number of items to include
 * @param {boolean} options.sortByValue - Whether to sort by value
 * @param {string} options.sortOrder - Sort order ('asc' or 'desc')
 * @returns {Array} Formatted chart data array
 */
export const prepareBarChartData = (items, options = {}) => {
  const { 
    nameKey = 'name', 
    valueKey = 'value',
    limit = 10,
    sortByValue = true,
    sortOrder = 'desc',
    maxNameLength = 20
  } = options;
  
  if (!items || !Array.isArray(items) || items.length === 0) {
    return [];
  }
  
  // Sort items if requested
  let sortedItems = [...items];
  
  if (sortByValue) {
    sortedItems.sort((a, b) => {
      const aValue = a[valueKey] || 0;
      const bValue = b[valueKey] || 0;
      
      return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
    });
  }
  
  // Limit number of items
  if (limit > 0) {
    sortedItems = sortedItems.slice(0, limit);
  }
  
  // Format item names (truncate if too long)
  return sortedItems.map(item => {
    const name = item[nameKey];
    const formattedName = name.length > maxNameLength 
      ? `${name.substring(0, maxNameLength)}...` 
      : name;
    
    return {
      name: formattedName,
      originalName: name,
      value: item[valueKey] || 0
    };
  });
};

/**
 * Generate chart colors from a base color
 * 
 * @param {string} baseColor - Base color in hex format
 * @param {number} count - Number of colors to generate
 * @returns {Array} Array of hex color strings
 */
export const generateChartColors = (baseColor = '#3b82f6', count = 5) => {
  // Convert hex to RGB
  const hex = baseColor.replace('#', '');
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  
  const colors = [];
  
  // Generate variants
  for (let i = 0; i < count; i++) {
    // Adjust lightness
    const factor = 0.7 + (i * 0.6 / count);
    
    const adjR = Math.min(255, Math.floor(r * factor));
    const adjG = Math.min(255, Math.floor(g * factor));
    const adjB = Math.min(255, Math.floor(b * factor));
    
    // Convert back to hex
    const hexR = adjR.toString(16).padStart(2, '0');
    const hexG = adjG.toString(16).padStart(2, '0');
    const hexB = adjB.toString(16).padStart(2, '0');
    
    colors.push(`#${hexR}${hexG}${hexB}`);
  }
  
  return colors;
};

/**
 * Common chart configurations that can be reused
 */
export const chartConfigs = {
  // Common tooltip formatter for revenue charts
  revenueTooltipFormatter: (value) => [formatCurrency(value), ''],
  
  // Common tooltip formatter for dates
  dateTooltipFormatter: (date) => formatDate(new Date(date), { format: 'long' }),
  
  // Common X-axis tick formatter for dates
  dateTickFormatter: (date) => {
    const dateObj = new Date(date);
    return `${dateObj.getMonth()+1}/${dateObj.getDate()}`;
  },
  
  // Common chart colors
  colors: {
    actual: '#3b82f6', // Blue
    predicted: '#10b981', // Green
    upperBound: '#d1d5db', // Light gray
    lowerBound: '#d1d5db', // Light gray
    positive: '#10b981', // Green
    negative: '#ef4444', // Red
    neutral: '#6b7280', // Gray
    // Category colors
    category1: '#8884d8', // Purple
    category2: '#82ca9d', // Green
    category3: '#ffc658', // Yellow
    category4: '#ff8042', // Orange
    category5: '#0088fe', // Blue
  }
};

/**
 * Determine if a chart should use a bar or line chart based on data characteristics
 * 
 * @param {Array} data - Data array
 * @param {Object} options - Additional options
 * @returns {string} Chart type ('bar', 'line', or 'area')
 */
export const determineChartType = (data, options = {}) => {
  const { 
    threshold = 30, // Number of points threshold
    timeSeriesKey = 'date'
  } = options;
  
  if (!data || !Array.isArray(data) || data.length === 0) {
    return 'bar'; // Default to bar
  }
  
  // Check if data is a time series
  const hasTimeSeries = data[0] && typeof data[0][timeSeriesKey] !== 'undefined';
  
  // Use line chart for time series data with many points
  if (hasTimeSeries && data.length > threshold) {
    return 'line';
  }
  
  // Use area chart for time series data with fewer points
  if (hasTimeSeries) {
    return 'area';
  }
  
  // Default to bar chart for non-time series data
  return 'bar';
};