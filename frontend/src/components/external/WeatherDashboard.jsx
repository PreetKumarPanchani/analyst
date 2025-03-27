import React, { useState, useEffect } from 'react';
import { 
  Cloud, Sun, CloudRain, CloudSnow, Wind, Thermometer, 
  Droplets, CloudSun, AlertTriangle, RefreshCw, Calendar
} from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchApi } from '../../utils/apiConfig';

const WeatherDashboard = () => {
  const [currentWeather, setCurrentWeather] = useState(null);
  const [forecast, setForecast] = useState([]);
  const [historical, setHistorical] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('current');

  useEffect(() => {
    fetchWeatherData();
  }, []);

  const fetchWeatherData = async () => {
    setLoading(true);
    try {
      // Fetch current weather
      const currentData = await fetchApi('/api/v1/external/weather/current');
      setCurrentWeather(currentData);
      
      // Fetch forecast
      const forecastData = await fetchApi('/api/v1/external/weather/forecast?days=7');
      setForecast(forecastData);
      
      // Fetch historical data (last 30 days)
      const today = new Date();
      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
      
      const startDate = thirtyDaysAgo.toISOString().split('T')[0];
      const endDate = today.toISOString().split('T')[0];
      
      const historicalData = await fetchApi(`/api/v1/external/weather/historical?start_date=${startDate}&end_date=${endDate}`);
      setHistorical(historicalData);
    } catch (err) {
      console.error("Error fetching weather data:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to get weather icon
  const getWeatherIcon = (weatherMain, size = 6) => {
    const iconClass = `w-${size} h-${size}`;
    
    switch(weatherMain?.toLowerCase()) {
      case 'clear':
      case 'sunny':
        return <Sun className={iconClass} />;
      case 'clouds':
      case 'cloudy':
        return <Cloud className={iconClass} />;
      case 'rain':
      case 'drizzle':
      case 'thunderstorm':
        return <CloudRain className={iconClass} />;
      case 'snow':
        return <CloudSnow className={iconClass} />;
      case 'mist':
      case 'fog':
      case 'haze':
        return <CloudSun className={iconClass} />;
      default:
        return <Cloud className={iconClass} />;
    }
  };

  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    });
  };

  // Prepare chart data
  const temperatureChartData = [...historical, ...forecast].map(day => ({
    date: day.date,
    min: day.min_temp,
    max: day.max_temp,
    avg: day.avg_temp,
    isHistorical: historical.some(h => h.date === day.date)
  }));

  const precipitationChartData = [...historical, ...forecast].map(day => ({
    date: day.date,
    precipitation: day.precipitation || 0,
    isHistorical: historical.some(h => h.date === day.date)
  }));

  if (loading && !currentWeather) {
    return (
      <div className="flex items-center justify-center p-12">
        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
        <span className="ml-2">Loading weather data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="bg-red-50 p-4 rounded-md border border-red-200">
          <div className="flex">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error loading weather data</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
          <button 
            onClick={fetchWeatherData}
            className="mt-4 flex items-center px-3 py-1.5 bg-red-100 text-red-700 rounded"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Header with tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex -mb-px">
          <button
            onClick={() => setActiveTab('current')}
            className={`py-4 px-6 font-medium text-sm border-b-2 ${
              activeTab === 'current'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Current Weather
          </button>
          <button
            onClick={() => setActiveTab('forecast')}
            className={`py-4 px-6 font-medium text-sm border-b-2 ${
              activeTab === 'forecast'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            7-Day Forecast
          </button>
          <button
            onClick={() => setActiveTab('historical')}
            className={`py-4 px-6 font-medium text-sm border-b-2 ${
              activeTab === 'historical'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Historical Data
          </button>
          <button
            onClick={() => setActiveTab('charts')}
            className={`py-4 px-6 font-medium text-sm border-b-2 ${
              activeTab === 'charts'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Charts
          </button>
        </nav>
      </div>

      {/* Current Weather Tab */}
      {activeTab === 'current' && currentWeather && (
        <div className="p-6">
          <div className="flex flex-col md:flex-row">
            <div className="flex-1 flex flex-col items-center justify-center mb-6 md:mb-0">
              <div className="text-5xl text-blue-500 mb-2">
                {getWeatherIcon(currentWeather.weather_main, 16)}
              </div>
              <h3 className="text-3xl font-bold text-gray-900 mb-1">{currentWeather.temperature.toFixed(1)}°C</h3>
              <p className="text-lg text-gray-600 capitalize">{currentWeather.weather_description}</p>
              <p className="text-sm text-gray-500">
                Feels like: {currentWeather.feels_like.toFixed(1)}°C
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Last updated: {new Date(currentWeather.timestamp).toLocaleTimeString()}
              </p>
            </div>
            
            <div className="flex-1">
              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center">
                  <Wind className="w-5 h-5 text-gray-500 mr-2" />
                  <div>
                    <p className="text-xs text-gray-500">Wind Speed</p>
                    <p className="text-sm font-medium text-gray-700">{currentWeather.wind_speed} m/s</p>
                  </div>
                </div>
                
                <div className="flex items-center">
                  <Droplets className="w-5 h-5 text-gray-500 mr-2" />
                  <div>
                    <p className="text-xs text-gray-500">Humidity</p>
                    <p className="text-sm font-medium text-gray-700">{currentWeather.humidity}%</p>
                  </div>
                </div>
                
                <div className="flex items-center">
                  <CloudRain className="w-5 h-5 text-gray-500 mr-2" />
                  <div>
                    <p className="text-xs text-gray-500">Precipitation (1h)</p>
                    <p className="text-sm font-medium text-gray-700">
                      {currentWeather.rain_1h || 0} mm
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center">
                  <Cloud className="w-5 h-5 text-gray-500 mr-2" />
                  <div>
                    <p className="text-xs text-gray-500">Cloud Cover</p>
                    <p className="text-sm font-medium text-gray-700">{currentWeather.clouds}%</p>
                  </div>
                </div>
              </div>
              
              <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-500 mb-2">Weather Impact on Sales</h4>
                <p className="text-sm text-gray-700">
                  {currentWeather.temperature > 25 
                    ? "Hot weather may increase sales of cold beverages and summer items."
                    : currentWeather.temperature < 10
                    ? "Cold weather typically increases demand for warm beverages and comfort foods."
                    : "Moderate temperatures have a neutral effect on most product categories."
                  }
                  {currentWeather.rain_1h > 0 
                    ? " Rainy conditions may reduce foot traffic but increase delivery orders."
                    : " Clear conditions typically result in higher foot traffic."
                  }
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Forecast Tab */}
      {activeTab === 'forecast' && forecast.length > 0 && (
        <div className="p-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
            {forecast.map((day, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50">
                <div className="text-center">
                  <p className="text-sm font-medium text-gray-900 mb-1">
                    {index === 0 ? 'Today' : formatDate(day.date)}
                  </p>
                  <div className="text-3xl text-blue-500 my-2">
                    {getWeatherIcon(day.weather_main, 10)}
                  </div>
                  <div className="flex justify-center items-center space-x-2 mb-2">
                    <span className="text-sm font-bold text-gray-900">{Math.round(day.max_temp)}°</span>
                    <span className="text-xs text-gray-500">/</span>
                    <span className="text-sm text-gray-500">{Math.round(day.min_temp)}°</span>
                  </div>
                  <p className="text-xs text-gray-500 capitalize">
                    {day.weather_main}
                  </p>
                  
                  <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
                    <div className="flex justify-between">
                      <span>Precip:</span>
                      <span>{day.precipitation || 0} mm</span>
                    </div>
                    <div className="flex justify-between mt-1">
                      <span>Humid:</span>
                      <span>{day.humidity}%</span>
                    </div>
                    <div className="flex justify-between mt-1">
                      <span>Wind:</span>
                      <span>{day.wind_speed} m/s</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-6 bg-blue-50 p-4 rounded-md border border-blue-200">
            <h4 className="text-sm font-medium text-blue-900 mb-1 flex items-center">
              <Calendar className="w-4 h-4 mr-1" />
              Forecast Impact on Sales
            </h4>
            <p className="text-sm text-blue-800">
              This forecast is used by our prediction models to adjust sales expectations. 
              {forecast.some(day => day.is_rainy)
                ? " Rainy days in the forecast suggest preparing for increased delivery and lower in-store traffic."
                : " Clear weather forecast indicates potential for higher foot traffic."}
              {forecast.some(day => day.max_temp > 25)
                ? " Hot days ahead recommend stocking up on seasonal items."
                : forecast.some(day => day.min_temp < 5)
                ? " Cold weather coming up suggests preparing winter inventory."
                : ""}
            </p>
          </div>
        </div>
      )}

      {/* Historical Data Tab */}
      {activeTab === 'historical' && historical.length > 0 && (
        <div className="p-6">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Weather
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Temperature
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Precipitation
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Humidity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Wind
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {historical.slice(0, 10).map((day, index) => (
                  <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatDate(day.date)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="text-blue-500 mr-2">
                          {getWeatherIcon(day.weather_main, 5)}
                        </div>
                        <span className="text-sm text-gray-900 capitalize">{day.weather_main}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">
                        {day.avg_temp.toFixed(1)}°C
                      </div>
                      <div className="text-xs text-gray-500">
                        {day.min_temp.toFixed(1)}° - {day.max_temp.toFixed(1)}°
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {day.precipitation || 0} mm
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {day.humidity}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {day.wind_speed} m/s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="mt-4 flex justify-center">
            <span className="text-sm text-gray-500">Showing 10 of {historical.length} days</span>
          </div>
        </div>
      )}

      {/* Charts Tab */}
      {activeTab === 'charts' && (
        <div className="p-6">
          <div className="mb-8">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Temperature Trends</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={temperatureChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => {
                      const date = new Date(value);
                      return `${date.getMonth()+1}/${date.getDate()}`;
                    }}
                  />
                  <YAxis 
                    yAxisId="left"
                    orientation="left"
                    label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    formatter={(value) => [`${Number(value).toFixed(1)}°C`, '']}
                    labelFormatter={(label) => formatDate(label)}
                  />
                  <Legend />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="max" 
                    name="Max Temp" 
                    stroke="#f59e0b" 
                    activeDot={(props) => {
                      const { dataKey, key, ...restProps } = props;
                      return <circle key={key} {...restProps} r={8} />;
                    }}
                    strokeWidth={2}
                  />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="avg" 
                    name="Avg Temp" 
                    stroke="#3b82f6" 
                    dot={(props) => {
                      const { dataKey, key, ...restProps } = props;
                      return <circle key={key} {...restProps} r={0} />;
                    }}
                    strokeWidth={2}
                  />
                  <Line 
                    yAxisId="left"
                    type="monotone" 
                    dataKey="min" 
                    name="Min Temp" 
                    stroke="#2dd4bf" 
                    activeDot={(props) => {
                      const { dataKey, key, ...restProps } = props;
                      return <circle key={key} {...restProps} r={8} />;
                    }}
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Precipitation</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={precipitationChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => {
                      const date = new Date(value);
                      return `${date.getMonth()+1}/${date.getDate()}`;
                    }}
                  />
                  <YAxis 
                    label={{ value: 'Precipitation (mm)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip
                    formatter={(value) => [`${Number(value).toFixed(1)} mm`, '']}
                    labelFormatter={(label) => formatDate(label)}
                  />
                  <Legend />
                  <Bar 
                    dataKey="precipitation" 
                    name="Precipitation" 
                    fill={(entry) => entry.isHistorical ? "#3b82f6" : "#93c5fd"}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/*
          <div className="mt-6 bg-gray-50 p-4 rounded-md border border-gray-200">
            <h4 className="text-sm font-medium text-gray-900 mb-1">Weather Impact on Forecast Models</h4>
            <p className="text-sm text-gray-700">
              Our sales forecasting models incorporate both historical and forecast weather data as external regressors.
              Temperature and precipitation patterns have shown significant correlation with sales trends across 
              different product categories.
            </p>
          </div>
          */}
        </div>
      )}
    </div>
  );
};

export default WeatherDashboard;