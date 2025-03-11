import React from 'react';

// Safely display any type of data
const DebugObject = ({ data, title = "Debug Data" }) => {
  const renderValue = (value) => {
    if (value === null) return "null";
    if (value === undefined) return "undefined";
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  return (
    <div className="p-3 border border-gray-300 bg-gray-50 rounded text-sm my-2">
      <h4 className="font-medium mb-1">{title}</h4>
      <pre className="overflow-auto max-h-[200px]">
        {renderValue(data)}
      </pre>
    </div>
  );
};

export default DebugObject; 