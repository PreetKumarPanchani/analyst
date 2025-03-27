/**
 * API Configuration
 * 
 * This file centralizes API URL configuration to make it easier to change
 * the API endpoint in one place.
 */

//export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://mm2xymkp2i.eu-west-2.awsapprunner.com/api/v1';


// Base API URL - detect if running locally and use local API when appropriate
const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';
export const API_URL = isLocalhost 
  ? 'http://localhost:8001/api/v1'  // Use local API when running on localhost
  : (process.env.NEXT_PUBLIC_API_URL || 'https://mm2xymkp2i.eu-west-2.awsapprunner.com/api/v1');

// For backward compatibility with code that uses relative URLs
export const getApiUrl = (endpoint) => {
  // If the endpoint already has the full URL, return it as is
  if (endpoint.startsWith('http')) {
    return endpoint;
  }
  
  // If endpoint already starts with /, trim it to avoid double slashes
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.substring(1) : endpoint;
  
  return `${API_URL}/${cleanEndpoint}`;
};

// Helper function to make API requests
export const fetchApi = async (endpoint, options = {}) => {
  const url = endpoint.startsWith('/api/v1') 
    ? `${API_URL}${endpoint.substring(7)}` // Remove /api/v1 prefix
    : getApiUrl(endpoint);
  
  // Add CORS headers when running locally
  if (isLocalhost) {
    options = {
      ...options,
      headers: {
        ...options.headers,
        'Content-Type': 'application/json',
      },
      mode: 'cors',
    };
  }
  
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
  } catch (error) {
    console.error(`Error fetching ${url}:`, error);
    throw error;
  }
};

export default {
  API_URL,
  getApiUrl,
  fetchApi
}; 