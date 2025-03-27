import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, AreaChart, Area, 
         XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Brush } from 'recharts';
import { ArrowLeft, RefreshCw, DownloadCloud, Info } from 'lucide-react';
import { fetchApi } from '../../utils/apiConfig';

const ProductForecastView = ({ company = 'forge', initialProduct = null }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(initialProduct);
  const [forecast, setForecast] = useState(null);
  const [generating, setGenerating] = useState(false);
  
  useEffect(() => {
    const fetchProducts = async () => {
      try {
        const data = await fetchApi(`/api/v1/sales/products/${company}`);
        setProducts(data);
        
        // If no initial product is selected, use the first one
        if (!initialProduct && data.length > 0) {
          setSelectedProduct(data[0]);
        }
      } catch (err) {
        console.error("Error fetching products:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    
    fetchProducts();
  }, [company, initialProduct]);
  
  useEffect(() => {
    if (selectedProduct) {
      fetchProductForecast(selectedProduct);
    }
  }, [selectedProduct]);
  
  const fetchProductForecast = async (product, forceRetrain = false) => {
    setGenerating(true);
    try {
      const queryParams = forceRetrain ? '?force_retrain=true' : '';
      const data = await fetchApi(`/api/v1/forecasts/product/${company}/${encodeURIComponent(product)}${queryParams}`);
      setForecast(data);
    } catch (err) {
      console.error("Error fetching product forecast:", err);
      setError(err.message);
    } finally {
      setGenerating(false);
    }
  };

  const handleProductChange = (e) => {
    setSelectedProduct(e.target.value);
  };
  
  const handleRegenerate = () => {
    fetchProductForecast(selectedProduct, true);
  };
  
  const prepareChartData = () => {
    if (!forecast || !forecast.dates) return [];
    
    return forecast.dates.map((date, index) => ({
      date,
      actual: forecast.actuals[index] || null,
      predicted: forecast.predictions[index],
      lowerBound: forecast.lower_bound[index],
      upperBound: forecast.upper_bound[index]
    }));
  };
  
  const chartData = prepareChartData();
  
  // Calculate historical vs forecast boundary index
  const historicalEndIndex = forecast?.actuals.findLastIndex(val => val !== null) || 0;
  
  // Calculate metrics
  const getMetrics = () => {
    if (!forecast) return { 
      mape: 0, 
      forecastQuantity: 0, 
      forecastPeriod: { start: '', end: '' } 
    };
    
    const mape = forecast.metrics?.mape || 0;
    
    // Sum forecasted quantities
    const forecastedData = forecast.predictions.slice(historicalEndIndex + 1);
    const forecastQuantity = forecastedData.reduce((sum, val) => sum + val, 0);
    
    // Get start and end dates of forecast period
    const forecastStartDate = new Date(forecast.dates[historicalEndIndex + 1]);
    const forecastEndDate = new Date(forecast.dates[forecast.dates.length - 1]);
    
    return {
      mape,
      forecastQuantity: forecastQuantity.toFixed(2),
      forecastPeriod: {
        start: forecastStartDate.toLocaleDateString(),
        end: forecastEndDate.toLocaleDateString()
      }
    };
  };
  
  const metrics = getMetrics();
  
  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
        <span className="ml-2">Loading product data...</span>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="bg-red-50 p-4 rounded-lg border border-red-200 max-w-md">
          <h3 className="text-red-800 font-medium">Error Loading Product Forecast</h3>
          <p className="text-red-700 mt-2">{error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="mt-4 bg-red-100 text-red-800 px-4 py-2 rounded hover:bg-red-200"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 pb-12">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <a href={`/dashboard/${company}`} className="flex items-center text-gray-500 hover:text-gray-700">
                <ArrowLeft className="w-5 h-5 mr-2" />
                Back to Dashboard
              </a>
              <h1 className="text-2xl font-bold text-gray-800 ml-8">Product Forecast</h1>
            </div>
            <div className="text-sm font-medium text-gray-500">
              Company: <span className="text-gray-900 capitalize">{company}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Controls */}
        <div className="bg-white p-6 rounded-lg shadow mb-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div className="w-full md:w-1/2 mb-4 md:mb-0">
              <label htmlFor="product-select" className="block text-sm font-medium text-gray-700 mb-1">
                Select Product
              </label>
              <select
                id="product-select"
                value={selectedProduct || ''}
                onChange={handleProductChange}
                className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
              >
                {products.map((product, index) => (
                  <option key={index} value={product}>
                    {product}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={handleRegenerate}
                disabled={generating}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300"
              >
                {generating ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Regenerate Forecast
                  </>
                )}
              </button>
              
              {/* <button
                className="flex items-center px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
              >
                <DownloadCloud className="w-4 h-4 mr-2" />
                Export
              </button>
              */}
            </div>

          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-blue-100 text-blue-600">
                <Info className="w-6 h-6" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Forecasted Quantity</p>
                <h3 className="text-xl font-bold text-gray-900">{metrics.forecastQuantity}</h3>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-green-100 text-green-600">
                <Info className="w-6 h-6" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Forecast Accuracy</p>
                <h3 className="text-xl font-bold text-gray-900">
                  {metrics.mape ? `${(100 - metrics.mape).toFixed(1)}%` : 'N/A'}
                </h3>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-purple-100 text-purple-600">
                <Info className="w-6 h-6" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Forecast Period</p>
                <h3 className="text-lg font-bold text-gray-900">
                  {`${metrics.forecastPeriod.start} - ${metrics.forecastPeriod.end}`}
                </h3>
              </div>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 gap-8">
          {/* Main Forecast Chart */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Quantity Forecast</h2>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => {
                      const date = new Date(value);
                      return `${date.getMonth()+1}/${date.getDate()}`;
                    }}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value) => [typeof value === 'number' ? value.toFixed(2) : value, '']}
                    labelFormatter={(label) => new Date(label).toLocaleDateString()}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="actual" 
                    stroke="#3b82f6" 
                    strokeWidth={2} 
                    dot={(props) => {
                      const { dataKey, key, ...restProps } = props;
                      return <circle key={key} {...restProps} r={3} />;
                    }}
                    activeDot={(props) => {
                      const { dataKey, key, ...restProps } = props;
                      return <circle key={key} {...restProps} r={6} />;
                    }}
                    name="Actual"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="predicted" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="Forecast" 
                    strokeDasharray={function(entry) {
                      // Use solid line for historical fitted values and dashed for future forecast
                      const index = chartData.indexOf(entry);
                      return index <= historicalEndIndex ? "0" : "5 5";
                    }}
                    /* dot={function(entry) {
                      const index = chartData.indexOf(entry.payload);
                      return index <= historicalEndIndex ? { r: 0 } : { r: 3 };
                    }}
                    */

                    // Render dots only for forecast period
                    dot={(props) => {
                      const index = chartData.indexOf(props.payload);
                      if (index <= historicalEndIndex) {
                        return null; // Don't render dots for historical fitted values
                      } else {
                        // Render dots only for forecast period
                        const { dataKey, key, ...restProps } = props;
                        return <circle key={key} {...restProps} r={3} />;
                      }
                    }}
                    
                  />
                  <Line 
                    type="monotone" 
                    dataKey="upperBound" 
                    stroke="#d1d5db"
                    strokeWidth={1}
                    name="Upper Bound"
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="lowerBound" 
                    stroke="#d1d5db" 
                    strokeWidth={1}
                    name="Lower Bound"
                    dot={false}
                  />
                  <Brush dataKey="date" height={30} stroke="#8884d8" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Components Chart (if available),  comment it out for now  */}
          {/*
          {forecast && forecast.components && (
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Forecast Components</h2>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="date" 
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => {
                        const date = new Date(value);
                        return `${date.getMonth()+1}/${date.getDate()}`;
                      }}
                    />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [typeof value === 'number' ? value.toFixed(2) : value, '']}
                      labelFormatter={(label) => new Date(label).toLocaleDateString()}
                    />
                    <Legend />
                    {forecast.components.trend && (
                      <Area 
                        type="monotone" 
                        dataKey={() => forecast.components.trend.map((v, i) => v)}
                        data={chartData}
                        fill="#8884d8" 
                        stroke="#8884d8"
                        name="Trend"
                        fillOpacity={0.3}
                      />
                    )}
                    {forecast.components.weekly && (
                      <Area 
                        type="monotone" 
                        dataKey={() => forecast.components.weekly.map((v, i) => v)}
                        data={chartData}
                        fill="#82ca9d" 
                        stroke="#82ca9d"
                        name="Weekly Pattern"
                        fillOpacity={0.3}
                      />
                    )}
                    {forecast.components.yearly && (
                      <Area 
                        type="monotone" 
                        dataKey={() => forecast.components.yearly.map((v, i) => v)}
                        data={chartData}
                        fill="#ffc658" 
                        stroke="#ffc658"
                        name="Yearly Pattern"
                        fillOpacity={0.3}
                      />
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
          */}
        </div>
      </main>
    </div>
  );
};

export default ProductForecastView;