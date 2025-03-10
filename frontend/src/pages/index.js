import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { BarChart2 } from 'lucide-react';

// Home page with redirect to dashboard
export default function Home() {
  const router = useRouter();
  
  // Redirect to forge dashboard on mount
  useEffect(() => {
    const timer = setTimeout(() => {
      router.push('/dashboard/forge');
    }, 1500);
    
    return () => clearTimeout(timer);
  }, [router]);
  
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100">
      <div className="w-20 h-20 flex items-center justify-center bg-blue-600 rounded-full mb-6">
        <BarChart2 className="w-10 h-10 text-white" />
      </div>
      <h1 className="text-3xl font-bold text-gray-900 mb-2">Sheffield Sales Forecast</h1>
      <p className="text-gray-600 mb-8">Analytics and forecasting for Sheffield businesses</p>
      <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
      <p className="mt-4 text-sm text-gray-500">Redirecting to dashboard...</p>
    </div>
  );
}