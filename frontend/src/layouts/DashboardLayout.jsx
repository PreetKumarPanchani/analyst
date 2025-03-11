import React from 'react';
import Navbar from '@/components/Navbar';
import Sidebar from '@/components/Sidebar';
import { useRouter } from 'next/router';
import ErrorBoundary from '@/components/ErrorBoundary';

export default function DashboardLayout({ children }) {
  const router = useRouter();
  
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      
      <div className="flex">
        <Sidebar />
        
        <main className="flex-1 p-4 md:p-6 lg:p-8">
          <div className="content-container">
            {/* Breadcrumbs */}
            <nav className="mb-4 text-sm text-gray-500">
              <ol className="flex items-center space-x-2">
                <li>Dashboard</li>
                {router.pathname.split('/').filter(Boolean).map((path, i) => (
                  <React.Fragment key={i}>
                    <li>/</li>
                    <li className="capitalize">{path}</li>
                  </React.Fragment>
                ))}
              </ol>
            </nav>
            
            {/* Main Content */}
            <div className="section-spacing">
              <ErrorBoundary>
                {children}
              </ErrorBoundary>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
} 