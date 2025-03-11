import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Package, 
  DollarSign, 
  Calendar, 
  ArrowRight, 
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw
} from 'lucide-react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ForecastInsights = ({ company }) => {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchInsights();
  }, [company]);

  const fetchInsights = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/v1/forecasts/insights/${company}`);
      if (!res.ok) throw new Error(`Failed to fetch insights: ${res.status}`);
      
      const data = await res.json();
      setInsights(data.insights);
    } catch (err) {
      console.error("Error fetching insights:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetchInsights();
    } finally {
      setRefreshing(false);
    }
  };

  // Prepare chart data for top products
  const prepareProductChartData = () => {
    if (!insights || !insights.product_insights) return [];
    
    return insights.product_insights.map(product => ({
      name: product.product.length > 15 
        ? product.product.substring(0, 15) + '...' 
        : product.product,
      quantity: product.expected_quantity,
      trend: product.growth_rate
    }));
  };

  if (loading && !insights) {
    return (
      <div className="bg-white shadow rounded-lg p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div className="h-24 bg-gray-200 rounded"></div>
            <div className="h-24 bg-gray-200 rounded"></div>
            <div className="h-24 bg-gray-200 rounded"></div>
          </div>
          <div className="h-64 bg-gray-200 rounded mb-6"></div>
          <div className="h-48 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white shadow rounded-lg p-6">
        <div className="bg-red-50 p-4 rounded-md border border-red-200">
          <div className="flex">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error Loading Insights</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
          <button 
            onClick={fetchInsights}
            className="mt-4 flex items-center px-3 py-1.5 bg-red-100 text-red-700 rounded"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!insights) {
    return (
      <div className="bg-white shadow rounded-lg p-6">
        <div className="text-center py-8">
          <AlertTriangle className="w-12 h-12 mx-auto text-yellow-500 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-1">No Insights Available</h3>
          <p className="text-gray-500 mb-4">
            We couldn't find any forecast insights for {company}.
          </p>
          <button
            onClick={handleRefresh}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center mx-auto"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Generate Insights
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white shadow rounded-lg">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center">
          <TrendingUp className="w-5 h-5 mr-2 text-blue-500" />
          Forecast Insights
        </h2>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300 flex items-center"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* Summary Cards */}
      <div className="p-6 border-b border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-blue-100 text-blue-600">
                <DollarSign className="w-6 h-6" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-blue-900">Expected Revenue</p>
                <h3 className="text-xl font-bold text-blue-900">
                  £{insights.expected_revenue.toLocaleString(undefined, {maximumFractionDigits: 2})}
                </h3>
                <p className="text-xs text-blue-700 mt-1 flex items-center">
                  {insights.revenue_trend === 'increasing' ? (
                    <>
                      <ArrowUpRight className="w-3 h-3 mr-1" />
                      Increasing trend
                    </>
                  ) : (
                    <>
                      <ArrowDownRight className="w-3 h-3 mr-1" />
                      Decreasing trend
                    </>
                  )}
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-purple-100 text-purple-600">
                <Calendar className="w-6 h-6" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-purple-900">Forecast Period</p>
                <h3 className="text-lg font-bold text-purple-900">
                  {new Date(insights.forecast_period.start).toLocaleDateString()} - {new Date(insights.forecast_period.end).toLocaleDateString()}
                </h3>
                <p className="text-xs text-purple-700 mt-1">
                  {Math.ceil((new Date(insights.forecast_period.end) - new Date(insights.forecast_period.start)) / (1000 * 60 * 60 * 24))} days
                </p>
              </div>
            </div>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-green-100 text-green-600">
                <Package className="w-6 h-6" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-green-900">Top Product</p>
                {insights.top_products && insights.top_products.length > 0 ? (
                  <>
                    <h3 className="text-md font-bold text-green-900 truncate max-w-xs">
                      {insights.top_products[0].product}
                    </h3>
                    <p className="text-xs text-green-700 mt-1">
                      {insights.top_products[0].total_quantity.toLocaleString()} units sold
                    </p>
                  </>
                ) : (
                  <h3 className="text-md font-bold text-green-900">No data available</h3>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Product Insights */}
      <div className="p-6 border-b border-gray-200">
        <h3 className="text-md font-semibold text-gray-900 mb-4">Product Performance Forecast</h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Product Forecast Chart */}
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={prepareProductChartData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(value) => [`${value.toLocaleString()}`, '']} />
                <Legend />
                <Bar 
                  dataKey="quantity" 
                  name="Forecasted Quantity" 
                  fill="#8884d8" 
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          {/* Product Growth Rate */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-3">Growth Rate by Product</h4>
            {insights.product_insights.map((product, index) => (
              <div key={index} className="mb-3">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-900 truncate max-w-xs">
                    {product.product}
                  </span>
                  <span 
                    className={`text-sm font-medium flex items-center ${
                      product.growth_rate > 0 
                        ? 'text-green-600' 
                        : product.growth_rate < 0 
                        ? 'text-red-600' 
                        : 'text-gray-600'
                    }`}
                  >
                    {product.growth_rate > 0 ? (
                      <ArrowUpRight className="w-3 h-3 mr-1" />
                    ) : product.growth_rate < 0 ? (
                      <ArrowDownRight className="w-3 h-3 mr-1" />
                    ) : null}
                    {product.growth_rate.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${
                      product.growth_rate > 0 
                        ? 'bg-green-500' 
                        : product.growth_rate < 0 
                        ? 'bg-red-500' 
                        : 'bg-gray-400'
                    }`}
                    style={{ 
                      width: `${Math.min(Math.abs(product.growth_rate * 2), 100)}%`, 
                      marginLeft: product.growth_rate < 0 ? 'auto' : '0' 
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Business Recommendations */}
      <div className="p-6">
        <h3 className="text-md font-semibold text-gray-900 mb-4">Business Recommendations</h3>
        
        <div className="space-y-4">
          {/* Revenue recommendation */}
          <div className="p-4 border border-gray-200 rounded-lg">
            <div className="flex items-start">
              <div className={`p-2 rounded-full ${
                insights.revenue_trend === 'increasing' 
                  ? 'bg-green-100 text-green-600' 
                  : 'bg-red-100 text-red-600'
              }`}>
                {insights.revenue_trend === 'increasing' 
                  ? <TrendingUp className="w-5 h-5" /> 
                  : <TrendingDown className="w-5 h-5" />
                }
              </div>
              <div className="ml-4">
                <h4 className="text-sm font-medium text-gray-900">Revenue Forecast</h4>
                <p className="text-sm text-gray-700 mt-1">
                  {insights.revenue_trend === 'increasing' 
                    ? `Revenue is trending upward with an expected total of £${insights.expected_revenue.toLocaleString(undefined, {maximumFractionDigits: 2})} for the forecast period. Consider increasing inventory levels to meet demand.`
                    : `Revenue is trending downward with an expected total of £${insights.expected_revenue.toLocaleString(undefined, {maximumFractionDigits: 2})} for the forecast period. Consider promotional activities to boost sales.`
                  }
                </p>
              </div>
            </div>
          </div>
          
          {/* Product mix recommendation */}
          <div className="p-4 border border-gray-200 rounded-lg">
            <div className="flex items-start">
              <div className="p-2 rounded-full bg-blue-100 text-blue-600">
                <Package className="w-5 h-5" />
              </div>
              <div className="ml-4">
                <h4 className="text-sm font-medium text-gray-900">Product Mix Optimization</h4>
                <p className="text-sm text-gray-700 mt-1">
                  {insights.product_insights.some(p => p.growth_rate > 10)
                    ? `Some products are showing strong growth trends. Focus inventory and marketing efforts on ${
                        insights.product_insights.filter(p => p.growth_rate > 10)[0].product
                      } which has a ${
                        insights.product_insights.filter(p => p.growth_rate > 10)[0].growth_rate.toFixed(1)
                      }% growth rate.`
                    : `No products are showing exceptional growth. Maintain balanced inventory across product categories.`
                  }
                </p>
              </div>
            </div>
          </div>
          
          {/* Timing recommendation */}
          <div className="p-4 border border-gray-200 rounded-lg">
            <div className="flex items-start">
              <div className="p-2 rounded-full bg-purple-100 text-purple-600">
                <Calendar className="w-5 h-5" />
              </div>
              <div className="ml-4">
                <h4 className="text-sm font-medium text-gray-900">Timing Considerations</h4>
                <p className="text-sm text-gray-700 mt-1">
                  The forecast period covers {Math.ceil((new Date(insights.forecast_period.end) - new Date(insights.forecast_period.start)) / (1000 * 60 * 60 * 24))} days, 
                  starting from {new Date(insights.forecast_period.start).toLocaleDateString()}. 
                  {insights.revenue_trend === 'increasing'
                    ? ' Plan for increased staffing levels during this period to accommodate higher customer traffic.'
                    : ' Consider adjusting staffing levels to optimize operational costs during this period.'
                  }
                </p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 text-center">
          <Link href={`/forecasts/${company}`} className="inline-flex items-center text-blue-600 hover:text-blue-800">
            View detailed forecasts
            <ArrowRight className="w-4 h-4 ml-1" />
          </Link>
        </div>
      </div>
    </div>
  );
};

// Add import for Link
import Link from 'next/link';

export default ForecastInsights;