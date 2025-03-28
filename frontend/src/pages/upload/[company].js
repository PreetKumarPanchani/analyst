import { useRouter } from 'next/router';
import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import AppLayout from '../../components/layout/AppLayout';
import { checkCompanyDataStatus } from '../../utils/apiConfig';

const UploadPage = () => {
  const router = useRouter();
  const { company } = router.query;
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [error, setError] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const fileInputRef = useRef(null);
  const [hasExistingData, setHasExistingData] = useState(false);
  const [mergeMode, setMergeMode] = useState('auto');
  const [checkingDataStatus, setCheckingDataStatus] = useState(true);

  // Validate company parameter
  useEffect(() => {
    if (company && !['forge', 'cpl'].includes(company)) {
      router.replace('/upload/forge');
    }
  }, [company, router]);

  useEffect(() => {
    if (!company) return;
    
    const checkDataStatus = async () => {
      setCheckingDataStatus(true);
      try {
        const exists = await checkCompanyDataStatus(company);
        setHasExistingData(exists);
      } catch (err) {
        console.error("Error checking data status:", err);
        // Default to assuming no data in case of error
        setHasExistingData(false);
      } finally {
        setCheckingDataStatus(false);
      }
    };
    
    checkDataStatus();
  }, [company]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      // Validate file type
      if (!selectedFile.name.endsWith('.xlsx') && !selectedFile.name.endsWith('.xls')) {
        setError('Please select an Excel file (.xlsx or .xls)');
        return;
      }
      
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file to upload');
      return;
    }
    
    try {
      setUploading(true);
      setUploadStatus('Uploading file...');
      
      // Create form data
      const formData = new FormData();
      formData.append('company', company);
      formData.append('file', file);
      
      // Add merge mode if provided
      if (mergeMode !== 'auto') {
        formData.append('merge_mode', mergeMode);
      }
      
      // Get the backend URL (defaulting to localhost in development)
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001/api/v1';
      
      // Upload the file directly to the backend
      const response = await fetch(`${backendUrl}/uploads/file`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header, browser will set it with the boundary
      });
      
      if (!response.ok) {
        // Try to parse error as JSON
        try {
          const errorData = await response.json();
          throw new Error(errorData.detail || `Upload failed with status: ${response.status}`);
        } catch (jsonError) {
          // If parsing as JSON fails, use status text
          throw new Error(`Upload failed with status: ${response.status} ${response.statusText}`);
        }
      }
      
      setUploadStatus('Processing file...');
      
      // Wait for processing to complete
      const result = await response.json();
      
      setHasExistingData(true); // Update data status after successful upload
      setUploadComplete(true);
      setUploadStatus('Upload and processing complete!');
    } catch (err) {
      console.error('Error uploading file:', err);
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setUploadComplete(false);
    setUploadStatus('');
    setError(null);
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  if (!company) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center p-12">
          <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
          <span className="ml-2">Loading...</span>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-6">
          <Link href={`/dashboard/${company}`} className="flex items-center text-gray-500 hover:text-gray-700 mb-2">
            {/* Back arrow */}
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 mr-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
            Back to Dashboard
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 capitalize">Upload {company} Data</h1>
          <p className="mt-2 text-sm text-gray-600">
            Upload new Excel transaction data for {company}. The system will process the file 
            and update the forecasts automatically.
          </p>
        </div>
        
        <div className="bg-white shadow rounded-lg">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">Upload Instructions</h2>
            <p className="mt-2 text-sm text-gray-600">
              The Excel file should contain transaction data in the standard format with sheets for:
            </p>
            <ul className="mt-2 ml-6 list-disc text-sm text-gray-600">
              <li>Sales</li>
              <li>Sales Items</li>
              <li>Sales Payments</li>
              <li>Sales Refunds</li>
              <li>Deleted Sales Items</li>
            </ul>
            {/* 
            <p className="mt-2 text-sm text-gray-600">
              After upload, the data will be processed, and the system will:
            </p>
            <ul className="mt-2 ml-6 list-disc text-sm text-gray-600">
              <li>Extract data from your Excel file</li>
              <li>Merge with existing data, adding only new transactions</li>
              <li>Update forecasts with the new data</li>
            </ul>
            */}
          </div>
          
          <div className="p-6">
            {!uploadComplete ? (
              <form onSubmit={handleSubmit} className="space-y-6">
                {hasExistingData && (
                  <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Data Handling
                    </label>
                    <div className="mt-2 bg-blue-50 p-4 rounded-md border border-blue-200">
                      <p className="text-sm text-blue-800 mb-3">
                        <strong> Data Upload Options </strong>
                      </p>
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center">
                          <input
                            id="merge-auto"
                            name="merge-mode"
                            type="radio"
                            value="auto"
                            checked={mergeMode === 'auto'}
                            onChange={() => setMergeMode('auto')}
                            className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                          />
                          <label htmlFor="merge-auto" className="ml-2 block text-sm text-gray-700">
                            Auto-merge
                          </label>
                        </div>
                        <div className="flex items-center">
                          <input
                            id="merge-replace"
                            name="merge-mode"
                            type="radio"
                            value="replace"
                            checked={mergeMode === 'replace'}
                            onChange={() => setMergeMode('replace')}
                            className="h-4 w-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                          />
                          <label htmlFor="merge-replace" className="ml-2 block text-sm text-gray-700">
                            Replace all data
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Upload Excel File
                  </label>
                  <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                    <div className="space-y-1 text-center">
                      <div className="flex text-sm text-gray-600">
                        <label htmlFor="file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500">
                          <span >Upload a file</span>
                          <input 
                            id="file-upload" 
                            name="file-upload" 
                            type="file" 
                            className="sr-only" 
                            onChange={handleFileChange}
                            ref={fileInputRef}
                            accept=".xlsx,.xls"
                          />
                        </label>
                        
                      </div>
                      <p className="text-xs text-gray-500">
                        Excel (.xlsx, .xls)
                      </p>
                    </div>
                  </div>
                </div>
                
                {file && (
                  <div className="flex items-center space-x-2 p-2 bg-blue-50 border border-blue-200 rounded">
                    {/* File icon */}
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                      <polyline points="14 2 14 8 20 8"></polyline>
                      <line x1="16" y1="13" x2="8" y2="13"></line>
                      <line x1="16" y1="17" x2="8" y2="17"></line>
                      <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                    <span className="text-sm text-blue-700">{file.name}</span>
                    <span className="text-xs text-gray-500">({Math.round(file.size / 1024)} KB)</span>
                  </div>
                )}
                
                {error && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-md">
                    <div className="flex">
                      {/* Alert icon */}
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-red-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                        <line x1="12" y1="9" x2="12" y2="13"></line>
                        <line x1="12" y1="17" x2="12.01" y2="17"></line>
                      </svg>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-red-800">{error}</p>
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => router.back()}
                    className="px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 mr-3"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={!file || uploading}
                    className="inline-flex justify-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-blue-300"
                  >
                    {uploading ? (
                      <>
                        {/* Spinner icon */}
                        <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Processing...
                      </>
                    ) : (
                      <>
                        {/* Upload icon */}
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                          <polyline points="17 8 12 3 7 8"></polyline>
                          <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        Upload File
                      </>
                    )}
                  </button>
                </div>
              </form>
            ) : (
              <div className="text-center py-8">
                {/* Check icon */}
                <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto mb-4 text-green-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                  <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
                <h3 className="text-lg font-medium text-gray-900">Upload Complete!</h3>
                <p className="mt-2 text-sm text-gray-600">
                  The file has been successfully uploaded and processed.
                </p>
                <div className="mt-6 flex justify-center space-x-4">
                  <button
                    onClick={resetUpload}
                    className="px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                  >
                    Upload Another File
                  </button>
                  <Link 
                    href={`/dashboard/${company}`}
                    className="px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
                  >
                    Go to Dashboard
                  </Link>
                </div>
              </div>
            )}
            
            {uploading && (
              <div className="mt-6">
                <div className="relative pt-1">
                  <div className="flex mb-2 items-center justify-between">
                    <div>
                      <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-blue-600 bg-blue-200">
                        {uploadStatus}
                      </span>
                    </div>
                  </div>
                  <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-gray-200">
                    <div className="w-full animate-pulse bg-blue-500"></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </AppLayout>
  );
};

export default UploadPage;