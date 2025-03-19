# API Configuration Guide

## Overview

This project uses a centralized API configuration to make it easier to manage API endpoints. The configuration is located in `src/utils/apiConfig.js`.

## How to Use

### Using the `fetchApi` Function (Recommended)

The simplest way to make API calls is to use the `fetchApi` function:

```javascript
import { fetchApi } from '../utils/apiConfig';

// Example usage
const fetchData = async () => {
  try {
    // You can use the direct endpoint path or keep the /api/v1 prefix
    const data = await fetchApi('/api/v1/sales/products/company-name');
    
    // Or use the endpoint without the prefix
    const data2 = await fetchApi('sales/products/company-name');
    
    // The function handles both formats
  } catch (error) {
    console.error('Error fetching data:', error);
  }
};
```

### Using the API_URL Directly

If you need to use the API URL directly:

```javascript
import { API_URL } from '../utils/apiConfig';

// Example usage
const fetchData = async () => {
  try {
    const response = await fetch(`${API_URL}/sales/products/company-name`);
    const data = await response.json();
  } catch (error) {
    console.error('Error fetching data:', error);
  }
};
```

### Using the getApiUrl Helper

For more complex URL construction:

```javascript
import { getApiUrl } from '../utils/apiConfig';

// Example usage
const url = getApiUrl(`sales/products/${companyName}?category=${category}`);
```

## Changing the API URL

To change the API URL, you can:

1. Set the `NEXT_PUBLIC_API_URL` environment variable
2. Or modify the default URL in `src/utils/apiConfig.js`

## Notes

- The API URL is centralized to avoid having to update it in multiple places
- The utility functions handle both relative and absolute URLs
- When using `fetchApi`, you can include or omit the `/api/v1` prefix, both will work 