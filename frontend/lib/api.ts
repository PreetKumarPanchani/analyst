/**
 * API client for the bakery analytics backend
 */

// Base URL for API
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api';

// Types
export interface SalesSummary {
  total_sales: number;
  transaction_count: number;
  avg_basket: number;
  start_date: string;
  end_date: string;
}

export interface MonthlySales {
  month: string;
  sales: number;
  transactions: number;
  avg_basket: number;
}

export interface DailySales {
  date: string;
  sales: number;
  transactions: number;
  avg_basket: number;
}

export interface RegisterSummary {
  register: string;
  sales: number;
  transactions: number;
  avg_basket: number;
}

export interface ProductSummary {
  product_name: string;
  category: string;
  sales: number;
  quantity: number;
}

export interface CategorySummary {
  category: string;
  sales: number;
  quantity: number;
  product_count: number;
}

export interface ForecastPoint {
  date: string;
  sales: number;
  sales_lower: number;
  sales_upper: number;
}

export interface ForecastResponse {
  forecast: ForecastPoint[];
  model_metrics: {
    mape: number;
    rmse: number;
    mae: number;
  };
}

export interface ForecastComponent {
  date: string;
  value: number;
  component: string;
}

export interface ForecastComponents {
  trend: ForecastComponent[];
  weekly: ForecastComponent[];
  yearly: ForecastComponent[];
}

/**
 * Generic fetch function with error handling
 */
async function fetchFromAPI<T>(endpoint: string, params: Record<string, any> = {}): Promise<T> {
  // Build URL with query parameters
  const url = new URL(`${API_BASE_URL}${endpoint}`);
  Object.keys(params).forEach(key => {
    if (params[key] !== undefined && params[key] !== null) {
      url.searchParams.append(key, params[key].toString());
    }
  });

  console.log('Fetching from URL:', url.toString());

  try {
    const response = await fetch(url.toString());
    
    if (!response.ok) {
      // Try to get error message from response
      let errorMessage = 'An error occurred';
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorData.message || errorMessage;
      } catch (e) {
        // If we can't parse the error, just use the status text
        errorMessage = response.statusText;
      }
      console.error('API Error:', response.status, errorMessage);
      throw new Error(`API error (${response.status}): ${errorMessage}`);
    }
    
    const data = await response.json();
    console.log('API Response:', data);
    return data as T;
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

// Sales API endpoints
export const salesApi = {
  getSummary: (company: string): Promise<SalesSummary> => {
    return fetchFromAPI<SalesSummary>('/sales/summary', { company });
  },
  
  getMonthlySales: (company: string): Promise<MonthlySales[]> => {
    return fetchFromAPI<MonthlySales[]>('/sales/monthly', { company });
  },
  
  getDailySales: (company: string, days: number = 30): Promise<DailySales[]> => {
    return fetchFromAPI<DailySales[]>('/sales/daily', { company, days });
  },
  
  getRegisterSummary: (company: string): Promise<RegisterSummary[]> => {
    return fetchFromAPI<RegisterSummary[]>('/sales/registers', { company });
  },
  
  getTopProducts: (company: string, limit: number = 10): Promise<ProductSummary[]> => {
    return fetchFromAPI<ProductSummary[]>('/sales/top-products', { company, limit });
  },
  
  getCategorySummary: (company: string): Promise<CategorySummary[]> => {
    return fetchFromAPI<CategorySummary[]>('/sales/categories', { company });
  }
};

// Forecast API endpoints
export const forecastApi = {
  getSalesForecast: (company: string, periods: number = 30): Promise<ForecastResponse> => {
    return fetchFromAPI<ForecastResponse>('/forecast/sales', { company, periods });
  },
  
  getComponents: (company: string): Promise<ForecastComponents> => {
    return fetchFromAPI<ForecastComponents>('/forecast/components', { company });
  }
};