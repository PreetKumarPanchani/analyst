// components/dashboard/TopProducts.tsx
import React from 'react';
import { formatCurrency } from '@/lib/utils';
import { ProductSummary } from '@/lib/api';

interface TopProductsProps {
  products: ProductSummary[];
  title?: string;
  loading?: boolean;
}

const TopProducts: React.FC<TopProductsProps> = ({ 
  products, 
  title = 'Top Products',
  loading = false
}) => {
  if (loading) {
    return (
      <div className="w-full bg-white rounded-lg shadow-md p-4">
        <h3 className="text-lg font-medium text-gray-700 mb-4">{title}</h3>
        <div className="animate-pulse">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex items-center py-3 border-b border-gray-100">
              <div className="w-8 h-8 bg-gray-200 rounded-full mr-3"></div>
              <div className="flex-1">
                <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                <div className="h-3 bg-gray-100 rounded w-1/4 mt-2"></div>
              </div>
              <div className="h-4 bg-gray-200 rounded w-16"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg shadow-md p-4">
      <h3 className="text-lg font-medium text-gray-700 mb-4">{title}</h3>
      <div className="overflow-hidden">
        {products.map((product, index) => (
          <div key={index} className="flex items-center py-3 border-b border-gray-100 last:border-b-0">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-medium mr-3">
              {index + 1}
            </div>
            <div className="flex-1">
              <h4 className="font-medium text-gray-800">{product.product_name}</h4>
              <p className="text-sm text-gray-500">{product.category} · {product.quantity} units</p>
            </div>
            <div className="text-right">
              <p className="font-medium text-gray-800">{formatCurrency(product.sales)}</p>
            </div>
          </div>
        ))}
        
        {products.length === 0 && (
          <div className="py-8 text-center text-gray-500">
            No product data available
          </div>
        )}
      </div>
    </div>
  );
};

export default TopProducts;