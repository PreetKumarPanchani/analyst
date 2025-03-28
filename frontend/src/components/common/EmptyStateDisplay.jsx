import React from 'react';
import { BarChart2, Upload, AlertCircle, TrendingUp, Package, ArrowLeft } from 'lucide-react';
import Link from 'next/link';

/**
 * Enhanced component to display when no data is available for charts or tables
 * 
 * @param {Object} props
 * @param {string} props.title - Main message title
 * @param {string} props.message - Detailed message
 * @param {string} props.type - Type of empty state ('noData', 'dataDeleted', 'error')
 * @param {string} props.company - Company identifier for navigation links
 * @param {boolean} props.showUploadLink - Whether to show upload link
 * @param {boolean} props.showDashboardLink - Whether to show return to dashboard link
 */
const EmptyStateDisplay = ({ 
  title = 'No Data Available', 
  message = 'There is currently no data to display.', 
  type = 'noData',
  company = 'forge',
  showUploadLink = true,
  showDashboardLink = true
}) => {
  // Select the appropriate icon based on the type
  const renderIcon = () => {
    switch (type) {
      case 'noData':
        return <TrendingUp className="w-16 h-16 text-gray-400 mx-auto mb-4" />;
      case 'dataDeleted':
        return <AlertCircle className="w-16 h-16 text-amber-400 mx-auto mb-4" />;
      case 'error':
        return <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />;
      default:
        return <Package className="w-16 h-16 text-gray-400 mx-auto mb-4" />;
    }
  };

  return (
    <div className="text-center py-12">
      {renderIcon()}
      <h2 className="text-2xl font-medium text-gray-900 mb-2">{title}</h2>
      <p className="mt-2 text-gray-500 max-w-xl mx-auto">{message}</p>
      
      {showUploadLink && (
        <div className="mt-6 space-x-4">
          <Link 
            href={`/upload/${company}`} 
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 transition-colors"
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload Data
          </Link>
          {showDashboardLink && (
            <Link 
              href={`/dashboard/${company}`} 
              className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 transition-colors"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Return to Dashboard
            </Link>
          )}
        </div>
      )}
    </div>
  );
};

export default EmptyStateDisplay;