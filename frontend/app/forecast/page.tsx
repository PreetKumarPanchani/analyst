'use client';

import React, { useState, useEffect } from 'react';
import { 
  salesApi, 
  forecastApi, 
  DailySales, 
  ForecastResponse,
  ForecastComponents 
} from '@/lib/api';
import { formatCurrency, formatNumber, getCompanyDisplayName } from '@/lib/utils';

// Components
import ForecastChart from '@/components/charts/ForecastChart';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function ForecastPage() {
  // State
  const [company, setCompany] = useState<string>('forge');
  const [forecastDays, setForecastDays] = useState<number>(30);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Data states
  const [dailySales, setDailySales] = useState<DailySales[]>([]);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [components, setComponents] = useState<ForecastComponents | null>(null);

  // Load data
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Load data in parallel
        const [dailySalesData, forecastData, componentsData] = await Promise.all([
          salesApi.getDailySales(company, 60), // Get 60 days of historical data
          forecastApi.getSalesForecast(company, forecastDays),
          forecastApi.getComponents(company)
        ]);
        
        // Update state
        setDailySales(dailySalesData);
        setForecast(forecastData);
        setComponents(componentsData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        console.error('Error loading forecast data:', err);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [company, forecastDays]);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Sales Forecast</h1>
          <p className="text-gray-500">Predicting future sales for {getCompanyDisplayName(company)}</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex space-x-2">
            <button
              onClick={() => setCompany('forge')}
              className={`px-4 py-2 rounded-md ${
                company === 'forge' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              Forge Bakehouse
            </button>
            <button
              onClick={() => setCompany('cpl')}
              className={`px-4 py-2 rounded-md ${
                company === 'cpl' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              CPLGPOPUP Ltd
            </button>
          </div>
          
          <select 
            value={forecastDays}
            onChange={(e) => setForecastDays(Number(e.target.value))}
            className="border border-gray-300 rounded-md p-2"
          >
            <option value={14}>14 days</option>
            <option value={30}>30 days</option>
            <option value={60}>60 days</option>
            <option value={90}>90 days</option>
          </select>
        </div>
      </div>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          <p>{error}</p>
        </div>
      )}
      
      {/* Main Forecast Chart */}
      <div className="mb-8">
        <ForecastChart
          historicalData={dailySales}
          forecastData={forecast?.forecast || []}
          title={`Sales Forecast (Next ${forecastDays} Days)`}
          loading={loading}
          height={500}
        />
      </div>
      
      {/* Forecast Metrics */}
      {!loading && forecast && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-sm font-medium text-gray-400">Mean Absolute Percentage Error</h3>
            <p className="text-2xl font-bold mt-1">
              {!isNaN(forecast.model_metrics.mape) 
                ? `${(forecast.model_metrics.mape * 100).toFixed(2)}%` 
                : 'N/A'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Lower values indicate better forecast accuracy
            </p>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-sm font-medium text-gray-400">Root Mean Square Error</h3>
            <p className="text-2xl font-bold mt-1">
              {!isNaN(forecast.model_metrics.rmse) 
                ? formatCurrency(forecast.model_metrics.rmse) 
                : 'N/A'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Average magnitude of forecast errors
            </p>
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-sm font-medium text-gray-400">Mean Absolute Error</h3>
            <p className="text-2xl font-bold mt-1">
              {!isNaN(forecast.model_metrics.mae) 
                ? formatCurrency(forecast.model_metrics.mae) 
                : 'N/A'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Average absolute difference between forecasts and actuals
            </p>
          </div>
        </div>
      )}
      
      {/* Forecast Components */}
      {!loading && components && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Trend Component */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-lg font-medium text-gray-700 mb-4">Sales Trend</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={components.trend}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return `${date.getDate()}/${date.getMonth() + 1}`;
                  }}
                />
                <YAxis 
                  tickFormatter={(value) => `£${value}`}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  formatter={(value) => [formatCurrency(Number(value)), 'Trend']}
                  labelFormatter={(label) => {
                    const date = new Date(label);
                    return date.toLocaleDateString('en-GB', { 
                      day: 'numeric', 
                      month: 'short', 
                      year: 'numeric' 
                    });
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  name="Trend"
                  stroke="#0088FE" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          {/* Weekly Seasonality */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-lg font-medium text-gray-700 mb-4">Weekly Pattern</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={components.weekly}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return new Intl.DateTimeFormat('en-GB', { weekday: 'short' }).format(date);
                  }}
                />
                <YAxis 
                  tickFormatter={(value) => `£${value}`}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  formatter={(value) => [`£${Number(value).toFixed(2)}`, 'Weekly Effect']}
                  labelFormatter={(label) => {
                    const date = new Date(label);
                    return date.toLocaleDateString('en-GB', { 
                      weekday: 'long',
                      day: 'numeric', 
                      month: 'short'
                    });
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  name="Weekly Effect"
                  stroke="#00C49F" 
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          {/* Yearly Seasonality (if available) */}
          {components.yearly && components.yearly.length > 0 && (
            <div className="bg-white rounded-lg shadow-md p-4 lg:col-span-2">
              <h3 className="text-lg font-medium text-gray-700 mb-4">Yearly Pattern</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={components.yearly}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => {
                      const date = new Date(value);
                      return new Intl.DateTimeFormat('en-GB', { month: 'short' }).format(date);
                    }}
                  />
                  <YAxis 
                    tickFormatter={(value) => `£${value}`}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip 
                    formatter={(value) => [`£${Number(value).toFixed(2)}`, 'Yearly Effect']}
                    labelFormatter={(label) => {
                      const date = new Date(label);
                      return date.toLocaleDateString('en-GB', { 
                        day: 'numeric', 
                        month: 'long'
                      });
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    name="Yearly Effect"
                    stroke="#FFBB28" 
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
      
      {/* Forecast Explanation */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-medium text-gray-700 mb-4">About The Forecast</h3>
        <div className="text-gray-600 space-y-4">
          <p>
            This forecast is generated using Facebook's Prophet algorithm, which is designed for time series 
            forecasting with strong seasonal patterns and holiday effects.
          </p>
          <p>
            The model takes into account:
          </p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Trend</strong>: The non-periodic changes in the time series</li>
            <li><strong>Weekly seasonality</strong>: Recurring patterns that happen on a weekly basis</li>
            <li><strong>Yearly seasonality</strong>: Annual patterns (requires at least 2 years of data)</li>
            <li><strong>Holidays</strong>: Special days that may affect sales</li>
          </ul>
          <p>
            The shaded area around the forecast line represents the 80% prediction interval - there is an 80% 
            chance that the actual value will fall within this range.
          </p>
        </div>
      </div>
    </div>
  );
}