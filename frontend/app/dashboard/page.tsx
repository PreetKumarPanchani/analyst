'use client';

import React, { useState, useEffect } from 'react';
import { 
  salesApi, 
  forecastApi, 
  SalesSummary, 
  MonthlySales,
  DailySales, 
  RegisterSummary, 
  ProductSummary, 
  CategorySummary, 
  ForecastResponse 
} from '@/lib/api';
import { formatCurrency, formatNumber, getCompanyDisplayName, calculatePercentChange } from '@/lib/utils';

// Components
import StatsCard from '@/components/ui/StatsCard';
import SalesChart from '@/components/charts/SalesChart';
import CategoryChart from '@/components/charts/CategoryChart';
import ForecastChart from '@/components/charts/ForecastChart';
import TopProducts from '@/components/dashboard/TopProducts';

export default function Dashboard() {
  // State
  const [company, setCompany] = useState<string>('forge');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  // Data states
  const [summary, setSummary] = useState<SalesSummary | null>(null);
  const [monthlySales, setMonthlySales] = useState<MonthlySales[]>([]);
  const [dailySales, setDailySales] = useState<DailySales[]>([]);
  const [registers, setRegisters] = useState<RegisterSummary[]>([]);
  const [topProducts, setTopProducts] = useState<ProductSummary[]>([]);
  const [categories, setCategories] = useState<CategorySummary[]>([]);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);

  // Calculate trends
  const calculateTrends = (data: MonthlySales[]) => {
    if (data.length < 2) return null;
    
    const current = data[data.length - 1];
    const previous = data[data.length - 2];
    
    return {
      sales: calculatePercentChange(current.sales, previous.sales),
      transactions: calculatePercentChange(current.transactions, previous.transactions),
      avgBasket: calculatePercentChange(current.avg_basket, previous.avg_basket),
    };
  };
  
  const trends = monthlySales.length >= 2 ? calculateTrends(monthlySales) : null;

  // Load data
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Load all data in parallel
        const [
          summaryData,
          monthlySalesData,
          dailySalesData,
          registersData,
          topProductsData,
          categoriesData,
          forecastData
        ] = await Promise.all([
          salesApi.getSummary(company),
          salesApi.getMonthlySales(company),
          salesApi.getDailySales(company, 30),
          salesApi.getRegisterSummary(company),
          salesApi.getTopProducts(company, 10),
          salesApi.getCategorySummary(company),
          forecastApi.getSalesForecast(company, 30)
        ]);
        
        // Update state
        setSummary(summaryData);
        setMonthlySales(monthlySalesData);
        setDailySales(dailySalesData);
        setRegisters(registersData);
        setTopProducts(topProductsData);
        setCategories(categoriesData);
        setForecast(forecastData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        console.error('Error loading data:', err);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [company]);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Bakery Analytics Dashboard</h1>
          <p className="text-gray-500">Analyzing sales data for {getCompanyDisplayName(company)}</p>
        </div>
        
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
      </div>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          <p>{error}</p>
        </div>
      )}
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <StatsCard
          title="Total Sales"
          value={summary ? formatCurrency(summary.total_sales) : '-'}
          trend={trends ? { value: trends.sales, label: 'vs previous month' } : undefined}
          loading={loading}
        />
        <StatsCard
          title="Total Transactions"
          value={summary ? formatNumber(summary.transaction_count) : '-'}
          trend={trends ? { value: trends.transactions, label: 'vs previous month' } : undefined}
          loading={loading}
        />
        <StatsCard
          title="Average Basket"
          value={summary ? formatCurrency(summary.avg_basket) : '-'}
          trend={trends ? { value: trends.avgBasket, label: 'vs previous month' } : undefined}
          loading={loading}
        />
      </div>
      
      {/* Sales Trend Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <SalesChart
          data={monthlySales}
          timeFrame="monthly"
          title="Monthly Sales"
          loading={loading}
        />
        <SalesChart
          data={dailySales}
          timeFrame="daily"
          title="Daily Sales (Last 30 Days)"
          loading={loading}
        />
      </div>
      
      {/* Forecast Chart */}
      <div className="mb-8">
        <ForecastChart
          historicalData={dailySales}
          forecastData={forecast?.forecast || []}
          title="Sales Forecast (Next 30 Days)"
          loading={loading}
        />
      </div>
      
      {/* Category and Products */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div className="lg:col-span-2">
          <CategoryChart
            data={categories}
            title="Sales by Category"
            loading={loading}
          />
        </div>
        <div>
          <TopProducts
            products={topProducts}
            title="Top Products"
            loading={loading}
          />
        </div>
      </div>
      
      {/* Register Performance */}
      <div className="mb-8">
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-medium text-gray-700 mb-4">Register Performance</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Register
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Sales
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Transactions
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Avg. Basket
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {registers.map((register, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {register.register}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(register.sales)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatNumber(register.transactions)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatCurrency(register.avg_basket)}
                    </td>
                  </tr>
                ))}
                
                {registers.length === 0 && !loading && (
                  <tr>
                    <td colSpan={4} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-center">
                      No register data available
                    </td>
                  </tr>
                )}
                
                {loading && (
                  <tr>
                    <td colSpan={4} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-center">
                      Loading...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}