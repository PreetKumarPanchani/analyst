/**
 * Utility functions for the bakery analytics frontend
 */

/**
 * Format a number as currency (GBP)
 */
export function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-GB', {
      style: 'currency',
      currency: 'GBP',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  }
  
  /**
   * Format a number with thousand separators
   */
  export function formatNumber(value: number): string {
    return new Intl.NumberFormat('en-GB').format(value);
  }
  
  /**
   * Format a date string to a more readable format
   */
  export function formatDate(dateString: string): string {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-GB', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    }).format(date);
  }
  
  /**
   * Format a month string (YYYY-MM) to a more readable format
   */
  export function formatMonth(monthString: string): string {
    const [year, month] = monthString.split('-');
    const date = new Date(parseInt(year), parseInt(month) - 1, 1);
    return new Intl.DateTimeFormat('en-GB', {
      month: 'short',
      year: 'numeric'
    }).format(date);
  }
  
  /**
   * Get a color for a given index (for charts)
   */
  export function getChartColor(index: number): string {
    const colors = [
      '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A28CFF',
      '#FF6B6B', '#4BC0C0', '#9FD356', '#F9A03F', '#59D4E8'
    ];
    return colors[index % colors.length];
  }
  
  /**
   * Calculate percentage change between two values
   */
  export function calculatePercentChange(current: number, previous: number): number {
    if (previous === 0) return 0;
    return ((current - previous) / Math.abs(previous)) * 100;
  }
  
  /**
   * Get a company's display name
   */
  export function getCompanyDisplayName(companyCode: string): string {
    const companies = {
      'forge': 'Forge Bakehouse',
      'cpl': 'CPLGPOPUP Ltd'
    };
    return companies[companyCode as keyof typeof companies] || companyCode;
  }
  
  /**
   * Group data by a key and calculate aggregations
   */
  export function groupBy<T>(
    data: T[], 
    keyFn: (item: T) => string, 
    aggregations: Record<string, (items: T[]) => number>
  ): Record<string, Record<string, number>> {
    const result: Record<string, Record<string, number>> = {};
    
    // Group items by key
    data.forEach(item => {
      const key = keyFn(item);
      if (!result[key]) {
        result[key] = {};
      }
    });
    
    // Apply aggregations to each group
    Object.keys(result).forEach(key => {
      const items = data.filter(item => keyFn(item) === key);
      Object.entries(aggregations).forEach(([name, fn]) => {
        result[key][name] = fn(items);
      });
    });
    
    return result;
  }
  
  /**
   * Generate an array of colors with gradient from start to end
   */
  export function generateColorGradient(startColor: string, endColor: string, steps: number): string[] {
    // Parse the hex colors to RGB
    const parseColor = (hexColor: string): [number, number, number] => {
      const r = parseInt(hexColor.slice(1, 3), 16);
      const g = parseInt(hexColor.slice(3, 5), 16);
      const b = parseInt(hexColor.slice(5, 7), 16);
      return [r, g, b];
    };
    
    const [startR, startG, startB] = parseColor(startColor);
    const [endR, endG, endB] = parseColor(endColor);
    
    // Calculate step size for each component
    const rStep = (endR - startR) / (steps - 1);
    const gStep = (endG - startG) / (steps - 1);
    const bStep = (endB - startB) / (steps - 1);
    
    // Generate the gradient colors
    return Array.from({ length: steps }, (_, i) => {
      const r = Math.round(startR + i * rStep);
      const g = Math.round(startG + i * gStep);
      const b = Math.round(startB + i * bStep);
      return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    });
  }
  
  /**
   * Get color based on value (for heatmaps)
   */
  export function getHeatmapColor(value: number, min: number, max: number): string {
    // Normalize value between 0 and 1
    const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
    
    // For blue heatmap (light blue to dark blue)
    const lightBlue = [224, 240, 255]; // #E0F0FF
    const darkBlue = [0, 84, 168];     // #0054A8
    
    // Interpolate colors
    const r = Math.round(lightBlue[0] + normalized * (darkBlue[0] - lightBlue[0]));
    const g = Math.round(lightBlue[1] + normalized * (darkBlue[1] - lightBlue[1]));
    const b = Math.round(lightBlue[2] + normalized * (darkBlue[2] - lightBlue[2]));
    
    return `rgb(${r}, ${g}, ${b})`;
  }