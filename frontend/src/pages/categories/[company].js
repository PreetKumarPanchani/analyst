import { useRouter } from 'next/router';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import AppLayout from '../../components/layout/AppLayout';
import LoadingSpinner from '../../components/common/LoadingSpinner';
import ErrorDisplay from '../../components/common/ErrorDisplay';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Calendar, TrendingUp, Package, ArrowRight } from 'lucide-react';

const CategoryPage = () => {
  const router = useRouter();
  const { company } = router.query;
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [categoryForecast, setCategoryForecast] = useState(null);
  const [products, setProducts] = useState([]);
  const [loadingForecast, setLoadingForecast] = useState(false);

  // Fetch categories when company param is available
  useEffect(() => {
    if (!company) return;

    if (!['forge', 'cpl'].includes(company)) {
      router.replace('/dashboard/forge');
      return;
    }

    const fetchCategories = async () => {
      try {
        setLoading(true);
        const res = await fetch(`/api/v1/sales/categories/${company}`);
        if (!res.ok) throw new Error(`Failed to fetch categories: ${res.status}`);
        
        const data = await res.json();
        setCategories(data);
        
        // Select first category by default
        if (data.length > 0) {
          setSelectedCategory(data[0]);
        }
      } catch (err) {
        console.error("Error fetching categories:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchCategories();
  }, [company, router]);

  // Fetch category forecast and products when a category is selected
  useEffect(() => {
    if (!company || !selectedCategory) return;

    const fetchCategoryData = async () => {
      setLoadingForecast(true);
      try {
        // Fetch forecast for selected category
        const forecastRes = await fetch(`/api/v1/forecasts/category/${company}/${encodeURIComponent(selectedCategory)}`);
        if (!forecastRes.ok) throw new Error(`Failed to fetch category forecast: ${forecastRes.status}`);
        
        const forecastData = await forecastRes.json();
        setCategoryForecast(forecastData);
        
        // Fetch products for selected category
        const productsRes = await fetch(`/api/v1/sales/products/${company}?category=${encodeURIComponent(selectedCategory)}`);
        if (!productsRes.ok) throw new Error(`Failed to fetch products: ${productsRes.status}`);
        
        const productsData = await productsRes.json();
        setProducts(productsData);
      } catch (err) {
        console.error("Error fetching category data:", err);
        // Don't set global error, just log it
      } finally {
        setLoadingForecast(false);
      }
    };

    fetchCategoryData();
  }, [company, selectedCategory]);

  const prepareChartData = () => {
    if (!categoryForecast || !categoryForecast.dates) return [];
    
    // Find the index where actuals end and forecasts begin
    const forecastStartIndex = categoryForecast.actuals.findLastIndex(val => val !== null) + 1;
    
    // Only include forecast period in chart
    return categoryForecast.dates.slice(forecastStartIndex).map((date, index) => ({
      date,
      forecast: categoryForecast.predictions[forecastStartIndex + index],
      lowerBound: categoryForecast.lower_bound[forecastStartIndex + index],
      upperBound: categoryForecast.upper_bound[forecastStartIndex + index]
    }));
  };

  if (loading) {
    return (
      <AppLayout>
        <LoadingSpinner message="Loading categories..." />
      </AppLayout>
    );
  }
  
  if (error) {
    return (
      <AppLayout>
        <ErrorDisplay 
          title="Error Loading Categories" 
          message={error}
          actionText="Go to Dashboard"
          actionHref={`/dashboard/${company}`}
        />
      </AppLayout>
    );
  }

  if (categories.length === 0) {
    return (
      <AppLayout>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center py-12">
            <Package className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-medium text-gray-900">No Categories Found</h2>
            <p className="mt-2 text-gray-500">There are no product categories available for {company}.</p>
            <Link href={`/dashboard/${company}`}>
              <a className="mt-6 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700">
                Return to Dashboard
              </a>
            </Link>
          </div>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900 capitalize">{company} Product Categories</h1>
          <p className="mt-2 text-sm text-gray-600">
            View and analyze product categories and their forecasted sales.
          </p>
        </div>
        
        {/* Category selection */}
        <div className="bg-white shadow rounded-lg mb-8">
          <div className="p-6">
            <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-2">
              Select Category
            </label>
            <select
              id="category"
              value={selectedCategory || ''}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="w-full md:w-1/2 p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
            >
              {categories.map((category, index) => (
                <option key={index} value={category}>{category}</option>
              ))}
            </select>
          </div>
        </div>
        
        {/* Selected category info */}
        {selectedCategory && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
            {/* Category forecast card */}
            <div className="bg-white shadow rounded-lg">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-900 mb-2">
                  {selectedCategory} Forecast
                </h2>
                {loadingForecast ? (
                  <div className="py-4 flex justify-center">
                    <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center py-4">
                    {categoryForecast && (
                      <>
                        <TrendingUp className="w-12 h-12 text-blue-500 mb-2" />
                        <div className="text-center">
                          <p className="text-3xl font-bold text-gray-900">
                            {categoryForecast && categoryForecast.predictions 
                              ? Math.round(categoryForecast.predictions.slice(-1)[0]) 
                              : '--'}
                          </p>
                          <p className="text-sm text-gray-500">
                            Forecasted units for next period
                          </p>
                        </div>
                        <Link href={`/forecasts/category/${company}/${encodeURIComponent(selectedCategory)}`}>
                          <a className="mt-6 flex items-center text-blue-600 hover:text-blue-800">
                            View detailed forecast
                            <ArrowRight className="w-4 h-4 ml-1" />
                          </a>
                        </Link>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
            
            {/* Products in category card */}
            <div className="bg-white shadow rounded-lg">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-900 mb-2">
                  Products in Category
                </h2>
                {loadingForecast ? (
                  <div className="py-4 flex justify-center">
                    <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                  </div>
                ) : (
                  <div className="py-4">
                    {products.length > 0 ? (
                      <ul className="divide-y divide-gray-200">
                        {products.slice(0, 5).map((product, index) => (
                          <li key={index} className="py-2">
                            <Link href={`/forecasts/product/${company}/${encodeURIComponent(product)}`}>
                              <a className="text-gray-900 hover:text-blue-600">{product}</a>
                            </Link>
                          </li>
                        ))}
                        {products.length > 5 && (
                          <li className="py-2 text-center">
                            <span className="text-sm text-gray-500">
                              +{products.length - 5} more products
                            </span>
                          </li>
                        )}
                      </ul>
                    ) : (
                      <p className="text-gray-500 text-center py-4">
                        No products found in this category
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
            
            {/* Sales trend card */}
            <div className="bg-white shadow rounded-lg">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-900 mb-2">
                  Seasonal Patterns
                </h2>
                {loadingForecast ? (
                  <div className="py-4 flex justify-center">
                    <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                  </div>
                ) : (
                  <div className="py-4">
                    {categoryForecast && categoryForecast.components && (
                      <>
                        <Calendar className="w-12 h-12 text-blue-500 mx-auto mb-2" />
                        <div className="text-center">
                          <p className="text-sm text-gray-500 mb-2">
                            {categoryForecast.components.weekly ? 'Weekly pattern detected' : 'No weekly pattern'}
                          </p>
                          <p className="text-sm text-gray-500">
                            {categoryForecast.components.yearly ? 'Yearly seasonality detected' : 'No yearly seasonality'}
                          </p>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        
        {/* Forecast chart */}
        {selectedCategory && categoryForecast && (
          <div className="bg-white shadow rounded-lg mb-8">
            <div className="p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">
                {selectedCategory} Forecast Chart
              </h2>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={prepareChartData()}>
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
                      labelFormatter={(label) => new Date(label).toLocaleDateString()}
                    />
                    <Legend />
                    <Bar 
                      dataKey="forecast" 
                      name="Forecast" 
                      fill="#8884d8" 
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
        
        {/* All categories table */}
        <div className="bg-white shadow rounded-lg">
          <div className="p-6">
            <h2 className="text-lg font-medium text-gray-900 mb-4">
              All Categories
            </h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Category Name
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Forecast
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {categories.map((category, index) => (
                    <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {category}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <Link href={`/forecasts/category/${company}/${encodeURIComponent(category)}`}>
                          <a className="text-blue-600 hover:text-blue-900">
                            View Forecast
                          </a>
                        </Link>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <button 
                          onClick={() => setSelectedCategory(category)}
                          className="text-gray-600 hover:text-blue-600"
                        >
                          Select
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  );
};

export default CategoryPage;