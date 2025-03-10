import React from 'react';
import AppLayout from '../../components/layout/AppLayout';
import WeatherDashboard from '../../components/external/WeatherDashboard';

const WeatherPage = () => {
  return (
    <AppLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Weather Insights</h1>
          <p className="mt-2 text-sm text-gray-600">
            Weather data is used as an external regressor in our forecasting models. View current conditions, 
            forecasts, and historical weather patterns that may affect sales.
          </p>
        </div>
        
        <div className="bg-white shadow rounded-lg mb-8">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">About Weather Forecasting</h2>
            <p className="mt-2 text-sm text-gray-600">
              Weather conditions have a significant impact on shopping behavior and sales patterns. Our forecasting 
              models incorporate weather data to improve prediction accuracy, especially for seasonal products.
            </p>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border border-blue-200 rounded-md p-4 bg-blue-50">
                <h3 className="font-medium text-blue-800">Temperature</h3>
                <p className="text-sm text-blue-700 mt-1">
                  Temperature affects demand for seasonal items, food and beverage choices, and shopping duration.
                </p>
              </div>
              <div className="border border-blue-200 rounded-md p-4 bg-blue-50">
                <h3 className="font-medium text-blue-800">Precipitation</h3>
                <p className="text-sm text-blue-700 mt-1">
                  Rain and snow can reduce foot traffic and increase online or delivery orders.
                </p>
              </div>
              <div className="border border-blue-200 rounded-md p-4 bg-blue-50">
                <h3 className="font-medium text-blue-800">Seasonality</h3>
                <p className="text-sm text-blue-700 mt-1">
                  Seasonal weather patterns help predict demand cycles for many product categories.
                </p>
              </div>
            </div>
          </div>
        </div>
        
        <WeatherDashboard />
      </div>
    </AppLayout>
  );
};

export default WeatherPage;