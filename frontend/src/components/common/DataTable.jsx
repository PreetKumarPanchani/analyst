import React, { useState } from 'react';
import { ArrowUpDown, ChevronDown, ChevronUp } from 'lucide-react';

/**
 * Reusable data table component with sorting and pagination
 * 
 * @param {Object} props
 * @param {Array} props.data - Array of data objects
 * @param {Array} props.columns - Column definitions [{ key, label, render, sortable }]
 * @param {number} props.initialPageSize - Initial page size (default: 10)
 * @param {string} props.initialSortKey - Initial sort key
 * @param {string} props.initialSortDir - Initial sort direction ('asc' or 'desc')
 * @param {boolean} props.paginated - Whether to enable pagination (default: true)
 * @param {string} props.emptyMessage - Message to display when no data available
 */
const DataTable = ({ 
  data = [],
  columns = [],
  initialPageSize = 10,
  initialSortKey = null,
  initialSortDir = 'asc',
  paginated = true,
  emptyMessage = 'No data available'
}) => {
  // Pagination state
  const [currentPage, setCurrentPage] = useState(0);
  const [pageSize, setPageSize] = useState(initialPageSize);
  
  // Sorting state
  const [sortKey, setSortKey] = useState(initialSortKey);
  const [sortDir, setSortDir] = useState(initialSortDir);
  
  // Calculate total pages
  const totalPages = paginated ? Math.ceil(data.length / pageSize) : 1;
  
  // Sort data
  const sortedData = [...data].sort((a, b) => {
    if (!sortKey) return 0;
    
    const valueA = a[sortKey];
    const valueB = b[sortKey];
    
    // Handle undefined or null values
    if (valueA === undefined || valueA === null) return sortDir === 'asc' ? -1 : 1;
    if (valueB === undefined || valueB === null) return sortDir === 'asc' ? 1 : -1;
    
    // Handle different data types
    if (typeof valueA === 'string' && typeof valueB === 'string') {
      return sortDir === 'asc' 
        ? valueA.localeCompare(valueB) 
        : valueB.localeCompare(valueA);
    }
    
    return sortDir === 'asc' ? valueA - valueB : valueB - valueA;
  });
  
  // Get current page data
  const currentData = paginated 
    ? sortedData.slice(currentPage * pageSize, (currentPage + 1) * pageSize)
    : sortedData;
  
  // Handle sort change
  const handleSort = (key) => {
    if (sortKey === key) {
      // Toggle direction if same key
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      // Set new key and default to ascending
      setSortKey(key);
      setSortDir('asc');
    }
    
    // Reset to first page when sorting changes
    setCurrentPage(0);
  };
  
  // Handle page size change
  const handlePageSizeChange = (e) => {
    const newSize = parseInt(e.target.value, 10);
    setPageSize(newSize);
    setCurrentPage(0); // Reset to first page
  };
  
  // Pagination controls
  const renderPagination = () => {
    if (!paginated || totalPages <= 1) return null;
    
    return (
      <div className="flex justify-between items-center mt-4">
        <div className="flex items-center">
          <span className="text-sm text-gray-700">
            Showing {currentData.length ? currentPage * pageSize + 1 : 0}-
            {Math.min((currentPage + 1) * pageSize, data.length)} of {data.length}
          </span>
          <select
            value={pageSize}
            onChange={handlePageSizeChange}
            className="ml-4 border border-gray-300 rounded px-2 py-1 text-sm"
          >
            <option value={5}>5 per page</option>
            <option value={10}>10 per page</option>
            <option value={25}>25 per page</option>
            <option value={50}>50 per page</option>
          </select>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setCurrentPage(0)}
            disabled={currentPage === 0}
            className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
          >
            First
          </button>
          <button
            onClick={() => setCurrentPage(currentPage - 1)}
            disabled={currentPage === 0}
            className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
          >
            Previous
          </button>
          <span className="px-3 py-1 text-sm">
            Page {currentPage + 1} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(currentPage + 1)}
            disabled={currentPage === totalPages - 1}
            className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
          >
            Next
          </button>
          <button
            onClick={() => setCurrentPage(totalPages - 1)}
            disabled={currentPage === totalPages - 1}
            className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
          >
            Last
          </button>
        </div>
      </div>
    );
  };
  
  // If no data available
  if (!data.length) {
    return (
      <div className="w-full overflow-hidden border border-gray-200 rounded-lg">
        <div className="p-6 text-center text-gray-500">{emptyMessage}</div>
      </div>
    );
  }
  
  return (
    <div className="w-full overflow-hidden border border-gray-200 rounded-lg">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columns.map((column, index) => (
                <th 
                  key={index}
                  className={`px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider ${
                    column.sortable ? 'cursor-pointer' : ''
                  }`}
                  onClick={() => column.sortable && handleSort(column.key)}
                >
                  <div className="flex items-center">
                    {column.label}
                    {column.sortable && sortKey === column.key && (
                      <span className="ml-1">
                        {sortDir === 'asc' ? (
                          <ChevronUp className="w-4 h-4" />
                        ) : (
                          <ChevronDown className="w-4 h-4" />
                        )}
                      </span>
                    )}
                    {column.sortable && sortKey !== column.key && (
                      <ArrowUpDown className="w-4 h-4 ml-1 text-gray-400" />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {currentData.map((row, rowIndex) => (
              <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                {columns.map((column, colIndex) => (
                  <td key={colIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {column.render 
                      ? column.render(row[column.key], row, rowIndex) 
                      : row[column.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {renderPagination()}
    </div>
  );
};

export default DataTable;