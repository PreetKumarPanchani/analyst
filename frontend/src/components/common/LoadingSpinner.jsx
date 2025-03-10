import React from 'react';

/**
 * Loading spinner component with customizable message
 * 
 * @param {Object} props
 * @param {string} props.message - Loading message to display
 * @param {string} props.size - Size of the spinner (small, medium, large)
 * @param {string} props.color - Color of the spinner
 */
const LoadingSpinner = ({ 
  message = 'Loading...', 
  size = 'medium',
  color = 'blue'
}) => {
  // Determine spinner size
  const spinnerSizes = {
    small: 'w-4 h-4 border-2',
    medium: 'w-8 h-8 border-4',
    large: 'w-12 h-12 border-4',
  };
  
  const spinnerSize = spinnerSizes[size] || spinnerSizes.medium;
  
  // Determine spinner color
  const spinnerColors = {
    blue: 'border-blue-500',
    gray: 'border-gray-500',
    green: 'border-green-500',
    red: 'border-red-500',
    yellow: 'border-yellow-500',
    purple: 'border-purple-500',
  };
  
  const spinnerColor = spinnerColors[color] || spinnerColors.blue;
  
  return (
    <div className="flex items-center justify-center p-12">
      <div className={`animate-spin ${spinnerSize} ${spinnerColor} border-t-transparent rounded-full`}></div>
      {message && <span className="ml-3 text-gray-700">{message}</span>}
    </div>
  );
};

export default LoadingSpinner;