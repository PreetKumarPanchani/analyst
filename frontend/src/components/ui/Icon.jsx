import React from 'react';
import { classNames } from '@/lib/utils';

// Standardized icon sizes with consistent proportions
const sizeClasses = {
  xs: 'w-3 h-3',
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-6 h-6',
  xl: 'w-7 h-7',
  '2xl': 'w-8 h-8'
};

/**
 * Consistent icon wrapper component
 * 
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - The icon element
 * @param {string} props.size - Size of the icon (xs, sm, md, lg, xl, 2xl)
 * @param {string} props.className - Additional classes
 */
export const Icon = ({ children, size = 'md', className = '', ...props }) => {
  return (
    <span 
      className={classNames(
        'inline-flex items-center justify-center',
        sizeClasses[size] || sizeClasses.md,
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};

export default Icon; 