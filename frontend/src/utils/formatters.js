/**
 * Utility functions for formatting various data types
 */

/**
 * Format a date value with various options
 * 
 * @param {Date|string} date - Date object or date string
 * @param {Object} options - Formatting options
 * @param {string} options.format - Format type: 'short', 'medium', 'long', 'full'
 * @returns {string} Formatted date string
 */
export const formatDate = (date, options = {}) => {
    const { format = 'medium' } = options;
    
    if (!date) return '';
    
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    
    // Check if date is valid
    if (isNaN(dateObj.getTime())) return 'Invalid date';
    
    switch (format) {
      case 'short':
        return dateObj.toLocaleDateString('en-GB', { 
          day: 'numeric', 
          month: 'short' 
        });
      case 'medium':
        return dateObj.toLocaleDateString('en-GB', { 
          day: 'numeric', 
          month: 'short', 
          year: 'numeric' 
        });
      case 'long':
        return dateObj.toLocaleDateString('en-GB', { 
          weekday: 'short',
          day: 'numeric', 
          month: 'long', 
          year: 'numeric' 
        });
      case 'full':
        return dateObj.toLocaleDateString('en-GB', { 
          weekday: 'long',
          day: 'numeric', 
          month: 'long', 
          year: 'numeric' 
        });
      case 'monthYear':
        return dateObj.toLocaleDateString('en-GB', { 
          month: 'long', 
          year: 'numeric' 
        });
      default:
        return dateObj.toLocaleDateString('en-GB');
    }
  };
  
  /**
   * Format a currency value
   * 
   * @param {number} amount - Numeric amount to format
   * @param {Object} options - Formatting options
   * @param {string} options.currency - Currency code (default: GBP)
   * @param {boolean} options.showSymbol - Whether to display the currency symbol
   * @param {number} options.decimals - Number of decimal places
   * @returns {string} Formatted currency string
   */
  export const formatCurrency = (amount, options = {}) => {
    const { 
      currency = 'GBP', 
      showSymbol = true, 
      decimals = 2 
    } = options;
    
    if (amount === null || amount === undefined) return '';
    
    // Format with Intl API
    const formatter = new Intl.NumberFormat('en-GB', {
      style: showSymbol ? 'currency' : 'decimal',
      currency: showSymbol ? currency : undefined,
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
    
    return formatter.format(amount);
  };
  
  /**
   * Format a number with thousands separators and decimal places
   * 
   * @param {number} value - Numeric value to format
   * @param {Object} options - Formatting options
   * @param {number} options.decimals - Number of decimal places
   * @param {boolean} options.compact - Whether to use compact notation (e.g., 1.2k)
   * @returns {string} Formatted number string
   */
  export const formatNumber = (value, options = {}) => {
    const { decimals = 0, compact = false } = options;
    
    if (value === null || value === undefined) return '';
    
    // Format with Intl API
    const formatter = new Intl.NumberFormat('en-GB', {
      notation: compact ? 'compact' : 'standard',
      compactDisplay: 'short',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
    
    return formatter.format(value);
  };
  
  /**
   * Format a percentage value
   * 
   * @param {number} value - Numeric value to format (0-100 or 0-1)
   * @param {Object} options - Formatting options
   * @param {number} options.decimals - Number of decimal places
   * @param {boolean} options.includeSymbol - Whether to include the % symbol
   * @returns {string} Formatted percentage string
   */
  export const formatPercent = (value, options = {}) => {
    const { decimals = 1, includeSymbol = true } = options;
    
    if (value === null || value === undefined) return '';
    
    // Check if value is in decimal form (0-1) and convert to percentage (0-100)
    const percentValue = value > 0 && value <= 1 ? value * 100 : value;
    
    // Format with Intl API
    const formatter = new Intl.NumberFormat('en-GB', {
      style: includeSymbol ? 'percent' : 'decimal',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
    
    // For percent style, divide by 100 because the formatter multiplies by 100
    return includeSymbol 
      ? formatter.format(percentValue / 100) 
      : formatter.format(percentValue);
  };
  
  /**
   * Truncate text to specified length with ellipsis
   * 
   * @param {string} text - Text to truncate
   * @param {number} length - Maximum length
   * @returns {string} Truncated text
   */
  export const truncateText = (text, length = 30) => {
    if (!text) return '';
    
    return text.length > length 
      ? `${text.substring(0, length)}...` 
      : text;
  };
  
  /**
   * Convert a JavaScript object to URL query parameters
   * 
   * @param {Object} params - Object with parameter key-value pairs
   * @returns {string} URL query string
   */
  export const objectToQueryParams = (params) => {
    if (!params || Object.keys(params).length === 0) return '';
    
    return Object.entries(params)
      .filter(([_, value]) => value !== undefined && value !== null && value !== '')
      .map(([key, value]) => {
        if (Array.isArray(value)) {
          return value
            .map(item => `${encodeURIComponent(key)}=${encodeURIComponent(item)}`)
            .join('&');
        }
        return `${encodeURIComponent(key)}=${encodeURIComponent(value)}`;
      })
      .join('&');
  };