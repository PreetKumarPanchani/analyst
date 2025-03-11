/**
 * Ensures data is safe to render in React components
 * @param {any} data - Data to be processed
 * @returns {any} - Safe-to-render data
 */
export const ensureRenderableProp = (data) => {
  if (data === null || data === undefined) return '';
  if (typeof data === 'object') return JSON.stringify(data);
  return data;
};

/**
 * Process an entire API response to ensure it's safe for rendering
 * @param {Object} response - API response object
 * @returns {Object} - Processed response safe for rendering
 */
export const processForecastResponse = (response) => {
  if (!response) return {};
  
  // Process components section which may contain chart objects
  if (response.components) {
    const components = {...response.components};
    Object.keys(components).forEach(key => {
      if (typeof components[key] === 'object' && components[key] !== null) {
        if (Array.isArray(components[key])) {
          components[key] = components[key].map(item => 
            typeof item === 'object' ? JSON.stringify(item) : item
          );
        } else {
          components[key] = JSON.stringify(components[key]);
        }
      }
    });
    return {...response, components};
  }
  
  return response;
}; 