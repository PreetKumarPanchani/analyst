import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

// Dummy data for demonstration purposes
const Dashboard = () => {
  const [companyData, setCompanyData] = useState('forge');
  const [timeFrame, setTimeFrame] = useState('monthly');
  const [forecastDays, setForecastDays] = useState(30);
  const [loading, setLoading] = useState(false);

  // Sample data - in a real app, this would come from the API
  const monthlySalesData = [
    { month: '2024-01', sales: 55330.45, transactions: 6356, avg_basket: 8.71 },
    { month: '2024-02', sales: 51832.33, transactions: 6420, avg_basket: 8.07 },
    { month: '2024-03', sales: 58000.12, transactions: 6800, avg_basket: 8.53 },
    { month: '2024-04', sales: 52500.78, transactions: 6250, avg_basket: 8.40 },
    { month: '2024-05', sales: 55100.34, transactions: 6470, avg_basket: 8.52 },
    { month: '2024-06', sales: 57200.45, transactions: 6700, avg_basket: 8.54 },
    { month: '2024-07', sales: 59800.23, transactions: 6920, avg_basket: 8.64 },
  ];
  
  const dailySalesData = Array.from({ length: 30 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (30 - i));
    return {
      date: date.toISOString().split('T')[0],
      sales: 1500 + Math.random() * 500,
      transactions: 180 + Math.random() * 50,
      avg_basket: 8 + Math.random() * 2
    };
  });
  
  const forecastData = Array.from({ length: forecastDays }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() + i + 1);
    const baseSales = 1700 + Math.random() * 600;
    return {
      date: date.toISOString().split('T')[0],
      sales: baseSales,
      sales_lower: baseSales * 0.85,
      sales_upper: baseSales * 1.15
    };
  });
  
  const categoryData = [
    { category: 'Pastry', sales: 40901.61, quantity: 12804, product_count: 75 },
    { category: 'Bread', sales: 26459.96, quantity: 8177, product_count: 29 },
    { category: 'Hot Drinks', sales: 16947.63, quantity: 5123, product_count: 24 },
    { category: 'Cafe Counter', sales: 7724.17, quantity: 1704, product_count: 10 },
    { category: 'Cold Drinks', sales: 858.92, quantity: 310, product_count: 11 },
  ];
  
  const topProducts = [
    { product_name: 'Flat White', category: 'Hot Drinks', sales: 4681.41, quantity: 1427 },
    { product_name: 'Pain au Chocolat', category: 'Pastry', sales: 3900.74, quantity: 1036 },
    { product_name: 'Latte', category: 'Hot Drinks', sales: 3974.97, quantity: 1129 },
    { product_name: 'Plain Croissant', category: 'Pastry', sales: 3014.44, quantity: 1067 },
    { product_name: 'White Peak Bread', category: 'Bread', sales: 3104.20, quantity: 730 },
  ];
  
  const registerSales = [
    { register: 'Train Station', sales: 42385.88, transactions: 4465, avg_basket: 9.49 },
    { register: 'Bakery Counter', sales: 38904.13, transactions: 4333, avg_basket: 8.98 },
    { register: 'Bakery Cafe', sales: 10013.24, transactions: 1598, avg_basket: 6.27 },
  ];
  
  // Format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP' }).format(value);
  };
  
  // Format number
  const formatNumber = (value) => {
    return new Intl.NumberFormat('en-GB').format(Math.round(value));
  };
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short' }).format(date);
  };
  
  // Format month
  const formatMonth = (monthString) => {
    const [year, month] = monthString.split('-');
    const date = new Date(parseInt(year), parseInt(month) - 1, 1);
    return new Intl.DateTimeFormat('en-GB', { month: 'short', year: 'numeric' }).format(date);
  };

  // Custom tooltip for sales charts
  const CustomTooltip = ({ active, payload, label, isMonthly }) => {
    if (active && payload && payload.length) {
      const dateLabel = isMonthly ? formatMonth(label) : formatDate(label);
      
      return (
        <div className="bg-white p-3 border border-gray-200 shadow-md rounded-md">
          <p className="font-medium">{dateLabel}</p>
          {payload.map((entry, index) => (
            <p key={`item-${index}`} style={{ color: entry.color }}>
              <span className="font-medium">{entry.name}: </span>
              {entry.dataKey === 'sales' || entry.dataKey === 'sales_lower' || entry.dataKey === 'sales_upper' || entry.dataKey === 'avg_basket'
                ? formatCurrency(entry.value) 
                : formatNumber(entry.value)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  // Chart colors
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  return (
    <div className="p-6 max-w-7xl mx-auto bg-gray-50">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Bakery Analytics Dashboard</h1>
          <p className="text-gray-500">Sales data for {companyData === 'forge' ? 'Forge Bakehouse' : 'CPLGPOPUP Ltd'}</p>
        </div>
        
        <div className="flex space-x-4">
          <div className="flex">
            <button
              onClick={() => setCompanyData('forge')}
              className={`px-4 py-2 rounded-l-md ${companyData === 'forge' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
            >
              Forge Bakehouse
            </button>
            <button
              onClick={() => setCompanyData('cpl')}
              className={`px-4 py-2 rounded-r-md ${companyData === 'cpl' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
            >
              CPLGPOPUP Ltd
            </button>
          </div>
          
          <div className="flex">
            <button
              onClick={() => setTimeFrame('monthly')}
              className={`px-4 py-2 rounded-l-md ${timeFrame === 'monthly' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
            >
              Monthly
            </button>
            <button
              onClick={() => setTimeFrame('daily')}
              className={`px-4 py-2 rounded-r-md ${timeFrame === 'daily' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
            >
              Daily
            </button>
          </div>
        </div>
      </div>
      
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="text-sm text-gray-500 mb-1">Total Sales</div>
          <div className="text-2xl font-bold">{formatCurrency(monthlySalesData.reduce((sum, item) => sum + item.sales, 0))}</div>
          <div className="text-sm text-green-500 mt-2">
            <span className="inline-block mr-1">↑</span>
            <span>4.2% vs previous period</span>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="text-sm text-gray-500 mb-1">Total Transactions</div>
          <div className="text-2xl font-bold">{formatNumber(monthlySalesData.reduce((sum, item) => sum + item.transactions, 0))}</div>
          <div className="text-sm text-green-500 mt-2">
            <span className="inline-block mr-1">↑</span>
            <span>3.1% vs previous period</span>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="text-sm text-gray-500 mb-1">Average Basket</div>
          <div className="text-2xl font-bold">{formatCurrency(
            monthlySalesData.reduce((sum, item) => sum + item.sales, 0) / 
            monthlySalesData.reduce((sum, item) => sum + item.transactions, 0)
          )}</div>
          <div className="text-sm text-green-500 mt-2">
            <span className="inline-block mr-1">↑</span>
            <span>1.8% vs previous period</span>
          </div>
        </div>
      </div>
      
      {/* Sales Trend Chart */}
      <div className="bg-white p-6 rounded-lg shadow-md mb-8">
        <h2 className="text-lg font-semibold text-gray-700 mb-4">Sales Trend</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={timeFrame === 'monthly' ? monthlySalesData : dailySalesData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey={timeFrame === 'monthly' ? "month" : "date"} 
                tickFormatter={timeFrame === 'monthly' ? 
                  (value) => formatMonth(value) : 
                  (value) => formatDate(value)
                }
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                tickFormatter={(value) => `£${Math.round(value / 1000)}k`}
                tick={{ fontSize: 12 }}
              />
              <Tooltip content={<CustomTooltip isMonthly={timeFrame === 'monthly'} />} />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="sales" 
                name="Sales"
                stroke="#0088FE" 
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Forecast Chart */}
      <div className="bg-white p-6 rounded-lg shadow-md mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-700">Sales Forecast</h2>
          <select 
            value={forecastDays} 
            onChange={(e) => setForecastDays(Number(e.target.value))}
            className="border border-gray-300 rounded-md p-2"
          >
            <option value={14}>14 Days</option>
            <option value={30}>30 Days</option>
            <option value={60}>60 Days</option>
            <option value={90}>90 Days</option>
          </select>
        </div>
        
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart
              data={[...dailySalesData.slice(-14), ...forecastData]}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="date" 
                tickFormatter={(value) => formatDate(value)}
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                tickFormatter={(value) => `£${Math.round(value / 1000)}k`}
                tick={{ fontSize: 12 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              
              {/* Historical Sales */}
              <Line 
                type="monotone" 
                dataKey="sales" 
                name="Historical Sales"
                stroke="#0088FE" 
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                connectNulls={true}
              />
              
              {/* Forecast confidence interval */}
              <Area 
                type="monotone" 
                dataKey="sales_upper" 
                name="Forecast Range" 
                stroke="none"
                fill="#82ca9d" 
                fillOpacity={0.2}
              />
              <Area 
                type="monotone" 
                dataKey="sales_lower" 
                stroke="none"
                fill="#82ca9d" 
                fillOpacity={0}
              />
              
              {/* Forecast line */}
              <Line 
                type="monotone" 
                dataKey="salesForecast" 
                name="Sales Forecast"
                stroke="#8884d8" 
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
                strokeDasharray="5 5"
                connectNulls={true}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      
      {/* Categories and Products */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Categories Chart */}
        <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-gray-700 mb-4">Sales by Category</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={categoryData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tickFormatter={(value) => `£${Math.round(value / 1000)}k`} />
                <YAxis 
                  dataKey="category" 
                  type="category" 
                  tick={{ fontSize: 12 }}
                  width={120}
                />
                <Tooltip 
                  formatter={(value) => [formatCurrency(value), 'Sales']}
                />
                <Legend />
                <Bar 
                  dataKey="sales" 
                  name="Sales"
                  fill="#0088FE" 
                  radius={[0, 4, 4, 0]} 
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Top Products */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-gray-700 mb-4">Top Products</h2>
          <div className="overflow-hidden">
            {topProducts.map((product, index) => (
              <div key={index} className="flex items-center py-3 border-b border-gray-100 last:border-b-0">
                <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-medium mr-3">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <h4 className="font-medium text-gray-800">{product.product_name}</h4>
                  <p className="text-sm text-gray-500">{product.category} · {formatNumber(product.quantity)} units</p>
                </div>
                <div className="text-right">
                  <p className="font-medium text-gray-800">{formatCurrency(product.sales)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Register Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Register Sales Chart */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-gray-700 mb-4">Register Sales</h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={registerSales}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="sales"
                  nameKey="register"
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}
                >
                  {registerSales.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [formatCurrency(value), 'Sales']} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Register Table */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-gray-700 mb-4">Register Details</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Register</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sales</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Transactions</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Avg. Basket</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {registerSales.map((register, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{register.register}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatCurrency(register.sales)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatNumber(register.transactions)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatCurrency(register.avg_basket)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;