import { useRouter } from 'next/router';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import AppLayout from '../../components/layout/AppLayout';
import LoadingSpinner from '../../components/common/LoadingSpinner';
import ErrorDisplay from '../../components/common/ErrorDisplay';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Calendar, Package, ShoppingBag, ArrowRight, RefreshCw } from 'lucide-react';
import { formatDate, formatCurrency } from '../../utils/formatters';
import api from '../../services/api';

const ForecastsPage = () => {
  const router = useRouter();
  const { company } = router.query;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [revenueForecasts, setRevenueForecasts] = useState(null);
  const [topProducts, setTopProducts] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [categoryForecast, setCategoryForecast] = useState(null);
  const [loadingCategory, setLoadingCategory] = useState(false);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    if (!company) return;

    if (!['forge', 'cpl'].includes(company)) {
      router.replace('/dashboard/forge');
      return;
    }

    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch revenue forecasts
        const revenueForecast = await api.forecast.getRevenueForecasts(company);
        setRevenueForecasts(revenueForecast);
        
        // Fetch top products
        const topProductsData = await api.sales.getTopProducts(company, 5);
        setTopProducts(topProductsData);
        
        // Fetch categories
        const categoriesData = await api.sales.getCategories(company);
        setCategories(categoriesData);
        
        // Select first category by default
        if (categoriesData.length > 0) {
          setSelectedCategory(categoriesData[0]);
          
          // Fetch forecast for first category
          setLoadingCategory(true);
          try {
            const categoryForecastData = await api.forecast.getCategoryForecast(company, categoriesData[0]);
            setCategoryForecast(categoryForecastData);
          } catch (err) {
            console.error("Error fetching category forecast:", err);
            // Don't set global error for this
          } finally {
            setLoadingCategory(false);
          }
        }
      } catch (err) {
        console.error("Error fetching forecast data:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [company, router]);

  // Handle category selection
  const handleCategorySelect = async (category) => {
    if (category === selectedCategory) return;
    
    setSelectedCategory(category);
    setLoadingCategory(true);
    
    try {
      const categoryForecastData = await api.forecast.getCategoryForecast(company, category);
      setCategoryForecast(categoryForecastData);
    } catch (err) {
      console.error("Error fetching category forecast:", err);
      // Just show empty state instead of error
      setCategoryForecast(null);
    } finally {
      setLoadingCategory(false);
    }
  };

  // Handle regenerate forecast
  const handleRegenerateForecast = async () => {
    try {
      setGenerating(true);
      
      // Force retrain the model
      const newForecast = await api.forecast.getRevenueForecasts(company, {
        forceRetrain: true
      });
      
      setRevenueForecasts(newForecast);
    } catch (err) {
      console.error("Error regenerating forecast:", err);
      // Show a toast or alert instead of setting global error
    } finally {
      setGenerating(false);
    }
  };

  // Prepare revenue chart data
  const prepareRevenueChartData = () => {
    if (!revenueForecasts || !revenueForecasts.dates) return [];
    
    // Find the index where actuals end and forecasts begin
    const forecastStartIndex = revenueForecasts.actuals.findLastIndex(val => val !== null) + 1;
    
    return revenueForecasts.dates.map((date, index) => ({
      date,
      actual: revenueForecasts.actuals[index] || null,
      predicted: revenueForecasts.predictions[index],
      upperBound: revenueForecasts.upper_bound?.[index] || null,
      lowerBound: revenueForecasts.lower_bound?.[index] || null,
      isForecasted: index >= forecastStartIndex
    }));
  };

  // Prepare category chart data
  const prepareCategoryChartData = () => {
    if (!categoryForecast || !categoryForecast.dates) return [];
    
    // Find the index where actuals end and forecasts begin
    const forecastStartIndex = categoryForecast.actuals.findLastIndex(val => val !== null) + 1;
    
    // Only include forecast portion
    return categoryForecast.dates.slice(forecastStartIndex).map((date, index) => ({
      date,
      forecast: categoryForecast.predictions[forecastStartIndex + index],
      category: selectedCategory
    }));
  };

  // Calculate metrics
  const getRevenueMetrics = () => {
    if (!revenueForecasts) return {
      totalRevenue: 0,
      accuracy: 0,
      forecastPeriod: { start: '', end: '' }
    };
    
    // Find the index where actuals end and forecasts begin
    const forecastStartIndex = revenueForecasts.actuals.findLastIndex(val => val !== null) + 1;
    
    // Sum forecasted revenue
    const forecastedData = revenueForecasts.predictions.slice(forecastStartIndex);
    const totalRevenue = forecastedData.reduce((sum, val) => sum + val, 0);
    
    // Get forecast accuracy (100 - MAPE)
    const accuracy = revenueForecasts.metrics?.mape 
      ? 100 - revenueForecasts.metrics.mape
      : null;
    
    // Get forecast period dates
    const startDate = new Date(revenueForecasts.dates[forecastStartIndex]);
    const endDate = new Date(revenueForecasts.dates[revenueForecasts.dates.length - 1]);
    
    return {
      totalRevenue,
      accuracy,
      forecastPeriod: {
        start: startDate,
        end: endDate
      }
    };
  };

  if (loading) {
    return (
      <AppLayout>
        <LoadingSpinner message="Loading forecasts..." />
      </AppLayout>
    );
  }
  
  if (error) {
    return (
      <AppLayout>
        <ErrorDisplay 
          title="Error Loading Forecasts" 
          message={error}
          actionText="Go to Dashboard"
          actionHref={`/dashboard/${company}`}
        />
      </AppLayout>
    );
  }

  const revenueMetrics = getRevenueMetrics();

  return (
    <AppLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 capitalize">{company} Forecasts</h1>
              <p className="mt-2 text-sm text-gray-600">
                View and analyze sales forecasts across different categories and products.
              </p>
            </div>
            
            <div className="mt-4 md:mt-0">
              <button
                onClick={handleRegenerateForecast}
                disabled={generating}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300"
              >
                {generating ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Regenerating...
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Regenerate Forecasts
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
        
        {/* Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-blue-100 text-blue-600">
                <TrendingUp className="w-6 h-6" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Total Forecasted Revenue</p>
                <h3 className="text-xl font-bold text-gray-900">
                  {formatCurrency(revenueMetrics.totalRevenue)}
                </h3>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-green-100 text-green-600">
                <Calendar className="w-6 h-6" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Forecast Period</p>
                <h3 className="text-lg font-bold text-gray-900">
                  {formatDate(revenueMetrics.forecastPeriod.start, { format: 'short' })} - {formatDate(revenueMetrics.forecastPeriod.end, { format: 'short' })}
                </h3>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-purple-100 text-purple-600">
                <Package className="w-6 h-6" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-500">Model Accuracy</p>
                <h3 className="text-xl font-bold text-gray-900">
                  {revenueMetrics.accuracy !== null 
                    ? `${revenueMetrics.accuracy.toFixed(1)}%` 
                    : 'N/A'}
                </h3>
              </div>
            </div>
          </div>
        </div>
        
        {/* Revenue Chart */}
        <div className="bg-white p-6 rounded-lg shadow mb-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Revenue Forecast</h2>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prepareRevenueChartData()}>
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
                  formatter={(value) => [formatCurrency(value), '']}
                  labelFormatter={(label) => formatDate(new Date(label), { format: 'long' })}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="actual" 
                  stroke="#3b82f6" 
                  strokeWidth={2} 
                  dot={(props) => {
                    if (props && props.payload && props.payload.actual !== null) {
                      const { dataKey, key, ...restProps } = props;
                      return <circle key={key} {...restProps} r={3} />;
                    } else {
                      return null;
                    }
                  }}
                  activeDot={(props) => {
                    const { dataKey, key, ...restProps } = props;
                    return <circle key={key} {...restProps} r={6} />;
                  }}
                  name="Actual Revenue"
                  isAnimationActive={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="predicted" 
                  stroke="#10b981" 
                  strokeWidth={2}
                  name="Forecast" 
                  dot={(props) => {
                    const chartData = prepareRevenueChartData();
                    const index = chartData.indexOf(props.payload);
                    const forecastStartIndex = revenueForecasts?.actuals.findLastIndex(val => val !== null) + 1;
                    
                    if (index >= forecastStartIndex) {
                      const { dataKey, key, ...restProps } = props;
                      return <circle key={key} {...restProps} r={3} />;
                    } else {
                      return null;
                    }
                  }}
                  strokeDasharray={function(entry) {
                    return entry && entry.payload && entry.payload.isForecasted ? "5 5" : "0";
                  }}
                  isAnimationActive={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="upperBound" 
                  stroke="#d1d5db" 
                  strokeWidth={1}
                  name="Upper Bound"
                  dot={false}
                  // Only display for forecasted period
                  connectNulls={true}
                  hide={revenueForecasts?.upper_bound === undefined}
                  strokeDasharray="3 3"
                />
                <Line 
                  type="monotone" 
                  dataKey="lowerBound" 
                  stroke="#d1d5db" 
                  strokeWidth={1}
                  name="Lower Bound"
                  dot={false}
                  // Only display for forecasted period
                  connectNulls={true}
                  hide={revenueForecasts?.lower_bound === undefined}
                  strokeDasharray="3 3"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 flex justify-center">
            <Link href={`/forecasts/revenue/${company}`} className="text-blue-600 hover:text-blue-800 flex items-center">
              View detailed revenue forecast
              <ArrowRight className="ml-1 w-4 h-4" />
            </Link>
          </div>
        </div>
        
        {/* Category Forecasts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Category Selector */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <ShoppingBag className="w-5 h-5 mr-2 text-blue-500" />
              Product Categories
            </h2>
            
            {categories.length > 0 ? (
              <div className="space-y-2">
                {categories.map((category, index) => (
                  <button
                    key={index}
                    onClick={() => handleCategorySelect(category)}
                    className={`w-full text-left px-4 py-2 rounded-md text-sm font-medium ${
                      selectedCategory === category
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-4">
                No categories available
              </p>
            )}
          </div>
          
          {/* Category Forecast */}
          <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              {selectedCategory ? `${selectedCategory} Forecast` : 'Category Forecast'}
            </h2>
            
            {loadingCategory ? (
              <div className="py-12 flex justify-center">
                <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
              </div>
            ) : categoryForecast ? (
              <>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={prepareCategoryChartData()}>
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
                        formatter={(value) => [Math.round(value), '']}
                        labelFormatter={(label) => formatDate(new Date(label), { format: 'long' })}
                      />
                      <Bar dataKey="forecast" name={`${selectedCategory} Forecast`} fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 flex justify-center">
                  <Link href={`/forecasts/category/${company}/${encodeURIComponent(selectedCategory)}`} className="text-blue-600 hover:text-blue-800 flex items-center">
                    View detailed {selectedCategory} forecast
                    <ArrowRight className="ml-1 w-4 h-4" />
                  </Link>
                </div>
              </>
            ) : (
              <div className="text-center py-12">
                <ShoppingBag className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900">No Forecast Available</h3>
                <p className="mt-2 text-gray-500">
                  {selectedCategory 
                    ? `No forecast data available for ${selectedCategory}`
                    : 'Select a category to view its forecast'}
                </p>
              </div>
            )}
          </div>
        </div>
        
        {/* Top Products */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Top Products Forecasts</h2>
          
          {topProducts.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Product Name
                    </th>
                    <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Units Sold
                    </th>
                    <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Revenue
                    </th>
                    <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {topProducts.map((product, index) => (
                    <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">
                        {product.product}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                        {product.total_quantity.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                        {formatCurrency(product.total_revenue)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-right">
                        <Link href={`/forecasts/product/${company}/${encodeURIComponent(product.product)}`} className="text-blue-600 hover:text-blue-800 inline-flex items-center">
                          View Forecast
                          <ArrowRight className="ml-1 h-4 w-4" />
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">
              No product data available
            </p>
          )}
          
          <div className="mt-6 flex justify-center">
            <Link href={`/products/${company}`} className="text-blue-600 hover:text-blue-800 flex items-center">
              View all products
              <ArrowRight className="ml-1 w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>
    </AppLayout>
  );
};

export default ForecastsPage;