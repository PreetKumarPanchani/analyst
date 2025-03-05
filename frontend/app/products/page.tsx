'use client';

import React, { useState, useEffect } from 'react';
import { 
  salesApi, 
  ProductSummary, 
  CategorySummary 
} from '@/lib/api';
import { formatCurrency, formatNumber, getCompanyDisplayName } from '@/lib/utils';

// Components
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

export default function ProductsPage() {
  // State
  const [company, setCompany] = useState<string>('forge');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<'list' | 'chart'>('list');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'sales' | 'quantity'>('sales');
  
  // Data states
  const [products, setProducts] = useState<ProductSummary[]>([]);
  const [categories, setCategories] = useState<CategorySummary[]>([]);

  // Load data
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Load data in parallel
        const [productsData, categoriesData] = await Promise.all([
          salesApi.getTopProducts(company, 100), // Get top 100 products
          salesApi.getCategorySummary(company)
        ]);
        
        // Update state
        setProducts(productsData);
        setCategories(categoriesData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        console.error('Error loading product data:', err);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [company]);

  // Filter and sort products
  const filteredProducts = products
    .filter(product => categoryFilter === 'all' || product.category === categoryFilter)
    .sort((a, b) => sortBy === 'sales' ? b.sales - a.sales : b.quantity - a.quantity);

  // Get unique categories for filter
  const uniqueCategories = Array.from(
    new Set(products.map(product => product.category))
  ).sort();

  // Chart colors
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1', '#a4de6c', '#d0ed57'];

  // Prepare data for category pie chart
  const categoryPieData = categories
    .sort((a, b) => b.sales - a.sales)
    .slice(0, 10)
    .map(category => ({
      name: category.category,
      value: category.sales
    }));

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Product Analysis</h1>
          <p className="text-gray-500">Analyzing product performance for {getCompanyDisplayName(company)}</p>
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
          
          <div className="flex space-x-2">
            <button
              onClick={() => setView('list')}
              className={`px-4 py-2 rounded-md ${
                view === 'list' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              List
            </button>
            <button
              onClick={() => setView('chart')}
              className={`px-4 py-2 rounded-md ${
                view === 'chart' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              Chart
            </button>
          </div>
        </div>
      </div>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          <p>{error}</p>
        </div>
      )}
      
      {/* Filters */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <div className="flex flex-wrap gap-4">
          <div>
            <label htmlFor="category-filter" className="block text-sm font-medium text-gray-700 mb-1">
              Category
            </label>
            <select
              id="category-filter"
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="border border-gray-300 rounded-md p-2 w-full"
            >
              <option value="all">All Categories</option>
              {uniqueCategories.map(category => (
                <option key={category} value={category}>{category}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label htmlFor="sort-by" className="block text-sm font-medium text-gray-700 mb-1">
              Sort By
            </label>
            <select
              id="sort-by"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'sales' | 'quantity')}
              className="border border-gray-300 rounded-md p-2 w-full"
            >
              <option value="sales">Sales (£)</option>
              <option value="quantity">Quantity</option>
            </select>
          </div>
        </div>
      </div>
      
      {/* Product List View */}
      {view === 'list' && (
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          {loading ? (
            <div className="p-8 text-center text-gray-500">Loading...</div>
          ) : filteredProducts.length === 0 ? (
            <div className="p-8 text-center text-gray-500">No products found</div>
          ) : (
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Rank
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Product
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Category
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Sales
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Avg Price
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredProducts.map((product, index) => (
                  <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {index + 1}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {product.product_name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {product.category}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatNumber(product.quantity)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(product.sales)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(product.sales / product.quantity)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
      
      {/* Product Chart View */}
      {view === 'chart' && !loading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Products Bar Chart */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-lg font-medium text-gray-700 mb-4">
              Top Products by {sortBy === 'sales' ? 'Sales' : 'Quantity'}
            </h3>
            <ResponsiveContainer width="100%" height={500}>
              <BarChart
                data={filteredProducts.slice(0, 15)}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  type="number"
                  tickFormatter={value => sortBy === 'sales' ? `£${value}` : value.toString()}
                />
                <YAxis 
                  type="category"
                  dataKey="product_name"
                  tick={{ fontSize: 12 }}
                  width={120}
                />
                <Tooltip 
                  formatter={(value, name) => [
                    sortBy === 'sales' ? formatCurrency(Number(value)) : formatNumber(Number(value)),
                    sortBy === 'sales' ? 'Sales' : 'Quantity'
                  ]}
                />
                <Legend />
                <Bar 
                  dataKey={sortBy} 
                  name={sortBy === 'sales' ? 'Sales' : 'Quantity'} 
                  fill="#0088FE"
                  radius={[0, 4, 4, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          {/* Category Distribution Pie Chart */}
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-lg font-medium text-gray-700 mb-4">
              Sales by Category
            </h3>
            <ResponsiveContainer width="100%" height={500}>
              <PieChart>
                <Pie
                  data={categoryPieData}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  outerRadius={160}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}
                >
                  {categoryPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value) => [formatCurrency(Number(value)), 'Sales']}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}