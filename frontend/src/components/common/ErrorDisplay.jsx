import React from 'react';
import Link from 'next/link';
import { AlertTriangle, RefreshCw } from 'lucide-react';

/**
 * Error display component for showing error messages to the user
 * 
 * @param {Object} props
 * @param {string} props.title - Error title
 * @param {string} props.message - Error message
 * @param {string} props.actionText - Text for action button
 * @param {string} props.actionHref - Link for action button
 * @param {Function} props.onRetry - Function to retry the operation
 */
const ErrorDisplay = ({ 
  title = 'Error', 
  message = 'Something went wrong', 
  actionText = 'Go Back',
  actionHref = '/',
  onRetry = null 
}) => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="bg-white shadow rounded-lg p-6 max-w-md mx-auto">
        <div className="flex items-center justify-center mb-4">
          <div className="bg-red-100 p-3 rounded-full">
            <AlertTriangle className="w-8 h-8 text-red-600" />
          </div>
        </div>
        <h2 className="text-xl font-semibold text-center text-gray-900 mb-2">{title}</h2>
        <p className="text-sm text-center text-gray-600 mb-6">{message}</p>
        <div className="flex flex-col sm:flex-row sm:justify-center space-y-3 sm:space-y-0 sm:space-x-4">
          <Link href={actionHref}>
            <a className="inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700">
              {actionText}
            </a>
          </Link>
          
          {onRetry && (
            <button 
              onClick={onRetry}
              className="inline-flex justify-center items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ErrorDisplay;