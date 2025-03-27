/**
 * API client for the Sheffield Sales Forecast application
 */

import { API_URL } from '../utils/apiConfig';

// Base API URL
//const API_BASE = '/api/v1';

const API_BASE = API_URL;

/**
 * Default fetch error handler
 */
const handleFetchError = async (response) => {
  if (!response.ok) {
    const error = new Error(`HTTP error! Status: ${response.status}`);
    
    try {
      const errorData = await response.json();
      error.data = errorData;
      error.message = errorData.detail || error.message;
    } catch (e) {
      // If error is not JSON, use status text
      error.message = response.statusText;
    }
    
    throw error;
  }
  
  return response;
};

/**
 * Generic API request function
 */
const apiRequest = async (endpoint, options = {}) => {
  // If endpoint starts with /, remove it to avoid double slashes
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;
  const url = `${API_BASE}/${cleanEndpoint}`;
  
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  const fetchOptions = {
    ...defaultOptions,
    ...options,
    headers: {
      ...defaultOptions.headers,
      ...options.headers,
    },
  };
  
  // For POST/PUT requests with JSON body
  if (
    (options.method === 'POST' || options.method === 'PUT') && 
    options.body && 
    typeof options.body !== 'string'
  ) {
    fetchOptions.body = JSON.stringify(options.body);
  }
  
  try {
    const response = await fetch(url, fetchOptions);
    await handleFetchError(response);
    
    // Check if response is empty
    const text = await response.text();
    return text ? JSON.parse(text) : {};
  } catch (error) {
    console.error(`API request failed: ${url}`, error);
    throw error;
  }
};

/**
 * API endpoints for sales data
 */
export const salesApi = {
  getCompanies: () => apiRequest('/sales/companies'),
  
  getCategories: (company) => apiRequest(`/sales/categories/${company}`),
  
  getProducts: (company, category = null) => {
    const url = category 
      ? `/sales/products/${company}?category=${encodeURIComponent(category)}`
      : `/sales/products/${company}`;
    return apiRequest(url);
  },
  
  getTopProducts: (company, limit = 10) => 
    apiRequest(`/sales/top-products/${company}?limit=${limit}`),
  
  processData: (company) => apiRequest(`/sales/process/${company}`, { method: 'POST' }),
};

/**
 * API endpoints for forecasts
 */
export const forecastApi = {
  getRevenueForecasts: (company, options = {}) => {
    const { periods = 15, includeWeather = true, includeEvents = true, forceRetrain = false } = options;
    
    const url = `/forecasts/revenue/${company}?periods=${periods}` +
      `&include_weather=${includeWeather}` +
      `&include_events=${includeEvents}` +
      `&force_retrain=${forceRetrain}`;
    
    return apiRequest(url);
  },
  
  getCategoryForecast: (company, category, options = {}) => {
    const { periods = 15, includeWeather = true, includeEvents = true, forceRetrain = false } = options;
    
    const url = `/forecasts/category/${company}/${encodeURIComponent(category)}?periods=${periods}` +
      `&include_weather=${includeWeather}` +
      `&include_events=${includeEvents}` +
      `&force_retrain=${forceRetrain}`;
    
    return apiRequest(url);
  },
  
  getProductForecast: (company, product, options = {}) => {
    const { periods = 15, includeWeather = true, includeEvents = true, forceRetrain = false } = options;
    
    const url = `/forecasts/product/${company}/${encodeURIComponent(product)}?periods=${periods}` +
      `&include_weather=${includeWeather}` +
      `&include_events=${includeEvents}` +
      `&force_retrain=${forceRetrain}`;
    
    return apiRequest(url);
  },
  
};

/**
 * API endpoints for events management
 */
export const eventsApi = {
  getEvents: (startDate = null, endDate = null) => {
    let url = '/external/events';
    
    if (startDate && endDate) {
      url += `?start_date=${startDate}&end_date=${endDate}`;
    } else if (startDate) {
      url += `?start_date=${startDate}`;
    } else if (endDate) {
      url += `?end_date=${endDate}`;
    }
    
    return apiRequest(url);
  },
  
  addEvent: (event) => apiRequest('/external/events', {
    method: 'POST',
    body: event,
  }),
  
  updateEvent: (eventName, eventDate, updatedData) => apiRequest(
    `/external/events/${encodeURIComponent(eventName)}?event_date=${eventDate}`,
    {
      method: 'PUT',
      body: updatedData,
    }
  ),
  
  deleteEvent: (eventName, eventDate) => apiRequest(
    `/external/events/${encodeURIComponent(eventName)}?event_date=${eventDate}`,
    {
      method: 'DELETE',
    }
  ),
};

/**
 * API endpoints for weather data
 */
export const weatherApi = {
  getCurrentWeather: () => apiRequest('/external/weather/current'),
  
  getForecast: (days = 7) => apiRequest(`/external/weather/forecast?days=${days}`),
  
  getHistoricalWeather: (startDate, endDate) => 
    apiRequest(`/external/weather/historical?start_date=${startDate}&end_date=${endDate}`),
};

// Export combined API object
const api = {
  sales: salesApi,
  forecast: forecastApi,
  events: eventsApi,
  weather: weatherApi,
};

export default api;