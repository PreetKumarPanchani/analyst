import { useEffect } from 'react';
import { useRouter } from 'next/router';
import LoadingSpinner from '../../components/common/LoadingSpinner';

// Redirect to default company page
export default function UploadIndexPage() {
  const router = useRouter();
  
  useEffect(() => {
    router.replace('/upload/forge');
  }, [router]);
  
  return (
    <div className="min-h-screen flex items-center justify-center">
      <LoadingSpinner message="Redirecting..." />
    </div>
  );
}