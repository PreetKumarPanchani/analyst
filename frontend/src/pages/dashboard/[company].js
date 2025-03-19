import { useRouter } from 'next/router';
import { useEffect } from 'react';
import AppLayout from '../../components/layout/AppLayout';
import SalesForecastDashboard from '../../components/dashboard/SalesForecastDashboard';
import ErrorDisplay from '../../components/common/ErrorDisplay';

const DashboardPage = () => {
  const router = useRouter();
  const { company } = router.query;
  
  // Validate company param
  useEffect(() => {
    if (company && !['forge', 'cpl'].includes(company)) {
      router.replace('/dashboard/forge');
    }
  }, [company, router]);
  
  if (!company) {
    return null; // Loading state
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