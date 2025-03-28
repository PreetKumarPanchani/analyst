import React, { useState } from 'react';
import { Trash2, RefreshCw, Upload, AlertTriangle, Check } from 'lucide-react';
import { deleteCompanyData } from '../../utils/apiConfig';

/**
 * Data Management Panel component for deleting or uploading company data
 * 
 * @param {Object} props
 * @param {string} props.company - Company identifier
 * @param {boolean} props.hasData - Whether the company has data
 * @param {Function} props.onDataChange - Callback when data is deleted or uploaded
 * @param {Function} props.onNavigateToUpload - Callback to navigate to upload page
 */
const DataManagementPanel = ({ 
  company, 
  hasData = false, 
  onDataChange,
  onNavigateToUpload
}) => {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState(null);
  const [deleteSuccess, setDeleteSuccess] = useState(false);

  const handleDeleteClick = () => {
    setShowConfirmation(true);
  };

  const handleCancelDelete = () => {
    setShowConfirmation(false);
  };

  const handleConfirmDelete = async () => {
    setIsDeleting(true);
    setDeleteError(null);
    setDeleteSuccess(false);
    
    try {
      const success = await deleteCompanyData(company);
      
      if (success) {
        setDeleteSuccess(true);
        if (onDataChange) {
          onDataChange(false); // Call the callback with hasData=false
        }
      } else {
        setDeleteError("Failed to delete data. Please try again.");
      }
    } catch (error) {
      setDeleteError(error.message || "An error occurred while deleting data.");
    } finally {
      setIsDeleting(false);
      // Don't hide confirmation yet, show the result
    }
  };

  return (
    <div className="bg-white shadow rounded-lg p-6">
      <h2 className="text-lg font-medium text-gray-900 mb-4">Data Management</h2>
      
      {!showConfirmation ? (
        <div className="flex flex-col space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">
                {hasData 
                  ? '' 
                  : 'No data available. Upload data to see forecasts.'}
              </p>
            </div>
            <div className="flex space-x-2">
              {hasData && (
                <button
                  onClick={handleDeleteClick}
                  className="flex items-center px-4 py-2 border border-red-300 text-red-700 rounded-md hover:bg-red-50"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Data
                </button>
              )}
              <button
                onClick={onNavigateToUpload}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload Data
              </button>
            </div>
          </div>
          
          <div className="text-sm">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${hasData ? 'bg-green-500' : 'bg-amber-500'}`}></div>
              <span>Data Status: {hasData ? 'Available' : 'Not Available'}</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {!deleteSuccess && !deleteError && (
            <>
              <div className="flex items-center p-3 bg-red-50 text-red-800 rounded-md">
                <AlertTriangle className="w-5 h-5 mr-2 flex-shrink-0" />
                <p className="text-sm">
                  Are you sure you want to delete all data for {company}? 
                  This action cannot be undone.
                </p>
              </div>
              
              <div className="flex justify-end space-x-3">
                <button
                  onClick={handleCancelDelete}
                  disabled={isDeleting}
                  className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleConfirmDelete}
                  disabled={isDeleting}
                  className="flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:bg-red-300"
                >
                  {isDeleting ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Deleting...
                    </>
                  ) : (
                    <>
                      <Trash2 className="w-4 h-4 mr-2" />
                      Delete Data
                    </>
                  )}
                </button>
              </div>
            </>
          )}
          
          {deleteSuccess && (
            <div className="flex items-center p-3 bg-green-50 text-green-800 rounded-md">
              <Check className="w-5 h-5 mr-2 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium">Data deleted successfully</p>
                <p className="text-xs mt-1">All data for {company} has been removed.</p>
                <button
                  onClick={() => setShowConfirmation(false)}
                  className="mt-2 text-sm underline"
                >
                  Close
                </button>
              </div>
            </div>
          )}
          
          {deleteError && (
            <div className="flex items-center p-3 bg-red-50 text-red-800 rounded-md">
              <AlertTriangle className="w-5 h-5 mr-2 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium">Error deleting data</p>
                <p className="text-xs mt-1">{deleteError}</p>
                <button
                  onClick={() => setShowConfirmation(false)}
                  className="mt-2 text-sm underline"
                >
                  Close
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DataManagementPanel;