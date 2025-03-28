import { useRouter } from 'next/router';
import { useState, useEffect } from 'react';
import Link from 'next/link';
import AppLayout from '../../components/layout/AppLayout';
import SalesForecastDashboard from '../../components/dashboard/SalesForecastDashboard';
import ErrorDisplay from '../../components/common/ErrorDisplay';
import { TrendingUp } from 'lucide-react';
import { checkCompanyDataStatus } from '../../utils/apiConfig';
import EmptyStateDisplay from '../../components/common/EmptyStateDisplay';

const DashboardPage = () => {
  const router = useRouter();
  const { company } = router.query;
  const [hasData, setHasData] = useState(true);
  const [loading, setLoading] = useState(true);
  
  // Validate company param
  useEffect(() => {
    if (company && !['forge', 'cpl'].includes(company)) {
      router.replace('/dashboard/forge');
    }
  }, [company, router]);
  
  // Check if data exists for this company
  useEffect(() => {
    if (!company) return;
    
    const checkDataStatus = async () => {
      try {
        const exists = await checkCompanyDataStatus(company);
        setHasData(exists);
      } catch (err) {
        console.error("Error checking data status:", err);
        // Default to assuming data exists in case of error
        setHasData(true);
      } finally {
        setLoading(false);
      }
    };
    
    checkDataStatus();
  }, [company]);
  
  if (!company || loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center p-12">
          <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
          <span className="ml-2">Loading...</span>
        </div>
      </AppLayout>
    );
  }
  
  if (!['forge', 'cpl'].includes(company)) {
    return (
      <AppLayout>
        <ErrorDisplay 
          title="Invalid Company" 
          message="Please select either Forge or CPL" 
          actionText="Go to Forge Dashboard"
          actionHref="/dashboard/forge"
        />
      </AppLayout>
    );
  }

  if (!hasData) {
    return (
      <AppLayout>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* <div className="text-center py-12">
            <TrendingUp className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-medium text-gray-900">No Data Available</h2>
            <p className="mt-2 text-gray-500">There is currently no data for {company}. Upload data to generate forecasts and analytics.</p>
            <div className="mt-6">
              <Link href={`/upload/${company}`} className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700">
                Upload Data
              </Link>
            </div>
            </div> */}

          <EmptyStateDisplay
            title="No Data Available"
            message={`There is currently no data for ${company}. Upload data to generate forecasts and analytics.`}
            type="noData"
            company={company}
            showUploadLink={true}
            showDashboardLink={false}
          />
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <SalesForecastDashboard selectedCompany={company} />
        </div>
        
      </div>
    </AppLayout>
  );

};

export default DashboardPage;