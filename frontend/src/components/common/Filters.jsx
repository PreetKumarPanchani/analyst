import React from 'react';
import { Search, Filter, RefreshCw, X } from 'lucide-react';

/**
 * Reusable filters component for data filtering
 * 
 * @param {Object} props
 * @param {Object} props.filters - Current filter values
 * @param {Function} props.onFilterChange - Function to handle filter changes
 * @param {Array} props.filterOptions - Available filter options
 * @param {boolean} props.showSearch - Whether to show search input
 * @param {string} props.searchPlaceholder - Placeholder for search input
 * @param {Function} props.onReset - Function to reset all filters
 */
const Filters = ({
  filters = {},
  onFilterChange,
  filterOptions = [],
  showSearch = true,
  searchPlaceholder = 'Search...',
  onReset
}) => {
  // Handle search input change
  const handleSearchChange = (e) => {
    onFilterChange({ ...filters, search: e.target.value });
  };
  
  // Handle filter change
  const handleFilterChange = (key, value) => {
    onFilterChange({ ...filters, [key]: value });
  };
  
  // Handle reset filters
  const handleReset = () => {
    if (onReset) {
      onReset();
    } else {
      // Default reset behavior - clear all filters
      const resetFilters = {};
      if (showSearch) resetFilters.search = '';
      onFilterChange(resetFilters);
    }
  };
  
  return (
    <div className="bg-white p-4 rounded-lg shadow mb-6">
      <div className="flex flex-col md:flex-row gap-4">
        {/* Search input */}
        {showSearch && (
          <div className="w-full md:w-1/3">
            <label htmlFor="search" className="block text-sm font-medium text-gray-700 mb-1">
              Search
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                id="search"
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                placeholder={searchPlaceholder}
                value={filters.search || ''}
                onChange={handleSearchChange}
              />
              {filters.search && (
                <button
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                  onClick={() => handleFilterChange('search', '')}
                >
                  <X className="h-4 w-4 text-gray-400" />
                </button>
              )}
            </div>
          </div>
        )}
        
        {/* Filter dropdowns */}
        {filterOptions.map((option, index) => (
          <div key={index} className="w-full md:w-auto flex-1">
            <label htmlFor={option.key} className="block text-sm font-medium text-gray-700 mb-1">
              {option.label}
            </label>
            <select
              id={option.key}
              className="block w-full py-2 pl-3 pr-10 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              value={filters[option.key] || ''}
              onChange={(e) => handleFilterChange(option.key, e.target.value)}
            >
              <option value="">{option.placeholder || 'All'}</option>
              {option.options.map((opt, idx) => (
                <option key={idx} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        ))}
        
        {/* Reset button */}
        <div className="w-full md:w-auto flex items-end">
          <button
            type="button"
            className="flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
            onClick={handleReset}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset Filters
          </button>
        </div>
      </div>
      
      {/* Active filters */}
      {Object.keys(filters).some(key => filters[key]) && (
        <div className="mt-4 flex flex-wrap gap-2">
          {Object.keys(filters).map(key => {
            if (!filters[key]) return null;
            
            // Find the label for the filter value
            let filterLabel = filters[key];
            const filterOption = filterOptions.find(opt => opt.key === key);
            if (filterOption) {
              const optionItem = filterOption.options.find(opt => opt.value === filters[key]);
              if (optionItem) {
                filterLabel = optionItem.label;
              }
            }
            
            return (
              <div 
                key={key}
                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
              >
                <span className="font-semibold mr-1">
                  {key === 'search' ? 'Search' : filterOptions.find(opt => opt.key === key)?.label || key}:
                </span>
                {filterLabel}
                <button
                  className="ml-1 focus:outline-none"
                  onClick={() => handleFilterChange(key, '')}
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default Filters;