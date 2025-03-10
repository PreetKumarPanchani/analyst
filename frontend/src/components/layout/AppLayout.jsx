import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { 
  BarChart2, 
  TrendingUp, 
  Package, 
  ShoppingBag, 
  Calendar, 
  Cloud, 
  Menu, 
  X, 
  ChevronDown,
  Settings,
  Home,
  LogOut
} from 'lucide-react';

const AppLayout = ({ children }) => {
  const router = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [companyDropdownOpen, setCompanyDropdownOpen] = useState(false);
  
  // Determine active company from URL or default to forge
  const getActiveCompany = () => {
    const path = router.asPath;
    if (path.includes('/cpl')) return 'cpl';
    return 'forge';
  };
  
  const activeCompany = getActiveCompany();

  const navigation = [
    { name: 'Dashboard', href: `/dashboard/${activeCompany}`, icon: Home },
    { name: 'Forecasts', href: `/forecasts/${activeCompany}`, icon: TrendingUp },
    { name: 'Products', href: `/products/${activeCompany}`, icon: Package },
    { name: 'Categories', href: `/categories/${activeCompany}`, icon: ShoppingBag },
    { name: 'Events', href: '/events', icon: Calendar },
    { name: 'Weather', href: '/weather', icon: Cloud },
  ];
  
  // Determine if a nav item is active
  const isActive = (href) => {
    const basePath = `/${router.asPath.split('/')[1]}`;
    return href.startsWith(basePath);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Mobile menu */}
      <div className="lg:hidden">
        <div className="fixed inset-0 flex z-40">
          {/* Mobile menu backdrop */}
          {mobileMenuOpen && (
            <div 
              className="fixed inset-0 bg-gray-600 bg-opacity-75"
              onClick={() => setMobileMenuOpen(false)}
            ></div>
          )}
          
          {/* Mobile menu sidebar */}
          <div 
            className={`
              fixed inset-y-0 left-0 flex flex-col w-64 bg-gray-800 transform transition duration-300 ease-in-out
              ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}
            `}
          >
            <div className="flex items-center justify-between h-16 px-4 border-b border-gray-700">
              <div className="flex items-center">
                <BarChart2 className="h-8 w-8 text-blue-400" />
                <span className="ml-2 text-xl font-semibold text-white">Sales Forecast</span>
              </div>
              <button 
                className="text-gray-400 hover:text-white"
                onClick={() => setMobileMenuOpen(false)}
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            
            <div className="px-2 py-4">
              {/* Company selector */}
              <div className="px-3 mb-4">
                <button
                  className="flex items-center justify-between w-full px-3 py-2 text-sm font-medium text-white bg-gray-700 rounded-md"
                  onClick={() => setCompanyDropdownOpen(!companyDropdownOpen)}
                >
                  <div className="flex items-center">
                    <ShoppingBag className="mr-2 h-5 w-5" />
                    <span className="capitalize">{activeCompany}</span>
                  </div>
                  <ChevronDown className={`ml-1 h-4 w-4 transform ${companyDropdownOpen ? 'rotate-180' : ''}`} />
                </button>
                
                {companyDropdownOpen && (
                  <div className="mt-1 bg-gray-700 rounded-md py-1">
                    <Link href={`/dashboard/forge`}>
                      <a 
                        className={`block px-4 py-2 text-sm ${activeCompany === 'forge' ? 'bg-gray-600 text-white' : 'text-gray-300 hover:bg-gray-600'}`}
                        onClick={() => {
                          setCompanyDropdownOpen(false);
                          setMobileMenuOpen(false);
                        }}
                      >
                        Forge
                      </a>
                    </Link>
                    <Link href={`/dashboard/cpl`}>
                      <a 
                        className={`block px-4 py-2 text-sm ${activeCompany === 'cpl' ? 'bg-gray-600 text-white' : 'text-gray-300 hover:bg-gray-600'}`}
                        onClick={() => {
                          setCompanyDropdownOpen(false);
                          setMobileMenuOpen(false);
                        }}
                      >
                        CPL
                      </a>
                    </Link>
                  </div>
                )}
              </div>
              
              {/* Navigation items */}
              <nav className="space-y-1">
                {navigation.map((item) => (
                  <Link key={item.name} href={item.href}>
                    <a
                      className={`
                        group flex items-center px-3 py-2 text-sm font-medium rounded-md
                        ${isActive(item.href) 
                          ? 'bg-gray-900 text-white' 
                          : 'text-gray-300 hover:bg-gray-700 hover:text-white'}
                      `}
                      onClick={() => setMobileMenuOpen(false)}
                    >
                      <item.icon 
                        className={`
                          mr-3 h-5 w-5 
                          ${isActive(item.href) ? 'text-blue-400' : 'text-gray-400 group-hover:text-gray-300'}
                        `} 
                      />
                      {item.name}
                    </a>
                  </Link>
                ))}
              </nav>
            </div>
            
            {/* Mobile menu footer */}
            <div className="mt-auto px-3 pb-3">
              <div className="pt-4 border-t border-gray-700">
                <Link href="/settings">
                  <a className="flex items-center px-3 py-2 text-sm font-medium text-gray-300 rounded-md hover:bg-gray-700 hover:text-white">
                    <Settings className="mr-3 h-5 w-5 text-gray-400" />
                    Settings
                  </a>
                </Link>
                <button className="mt-1 flex items-center px-3 py-2 text-sm font-medium text-gray-300 rounded-md hover:bg-gray-700 hover:text-white w-full">
                  <LogOut className="mr-3 h-5 w-5 text-gray-400" />
                  Sign out
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:flex lg:fixed lg:inset-y-0 lg:flex-col lg:w-64 lg:z-50">
        <div className="flex flex-col flex-1 min-h-0 bg-gray-800">
          <div className="flex items-center h-16 px-4 border-b border-gray-700">
            <BarChart2 className="h-8 w-8 text-blue-400" />
            <span className="ml-2 text-xl font-semibold text-white">Sales Forecast</span>
          </div>
          
          <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
            {/* Company selector */}
            <div className="px-3 mb-6">
              <button
                className="flex items-center justify-between w-full px-3 py-2 text-sm font-medium text-white bg-gray-700 rounded-md"
                onClick={() => setCompanyDropdownOpen(!companyDropdownOpen)}
              >
                <div className="flex items-center">
                  <ShoppingBag className="mr-2 h-5 w-5" />
                  <span className="capitalize">{activeCompany}</span>
                </div>
                <ChevronDown className={`ml-1 h-4 w-4 transform ${companyDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              
              {companyDropdownOpen && (
                <div className="mt-1 bg-gray-700 rounded-md py-1 absolute z-10 w-52">
                  <Link href={`/dashboard/forge`}>
                    <a 
                      className={`block px-4 py-2 text-sm ${activeCompany === 'forge' ? 'bg-gray-600 text-white' : 'text-gray-300 hover:bg-gray-600'}`}
                      onClick={() => setCompanyDropdownOpen(false)}
                    >
                      Forge
                    </a>
                  </Link>
                  <Link href={`/dashboard/cpl`}>
                    <a 
                      className={`block px-4 py-2 text-sm ${activeCompany === 'cpl' ? 'bg-gray-600 text-white' : 'text-gray-300 hover:bg-gray-600'}`}
                      onClick={() => setCompanyDropdownOpen(false)}
                    >
                      CPL
                    </a>
                  </Link>
                </div>
              )}
            </div>
            
            {/* Navigation */}
            <nav className="px-3 space-y-1">
              {navigation.map((item) => (
                <Link key={item.name} href={item.href}>
                  <a
                    className={`
                      group flex items-center px-3 py-2 text-sm font-medium rounded-md 
                      ${isActive(item.href) 
                        ? 'bg-gray-900 text-white' 
                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'}
                    `}
                  >
                    <item.icon 
                      className={`
                        mr-3 h-5 w-5 
                        ${isActive(item.href) ? 'text-blue-400' : 'text-gray-400 group-hover:text-gray-300'}
                      `} 
                    />
                    {item.name}
                  </a>
                </Link>
              ))}
            </nav>
          </div>
          
          {/* Sidebar footer */}
          <div className="p-3 border-t border-gray-700">
            <Link href="/settings">
              <a className="flex items-center px-3 py-2 text-sm font-medium text-gray-300 rounded-md hover:bg-gray-700 hover:text-white">
                <Settings className="mr-3 h-5 w-5 text-gray-400" />
                Settings
              </a>
            </Link>
            <button className="mt-1 flex items-center px-3 py-2 text-sm font-medium text-gray-300 rounded-md hover:bg-gray-700 hover:text-white w-full">
              <LogOut className="mr-3 h-5 w-5 text-gray-400" />
              Sign out
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64 flex flex-col">
        {/* Top navigation */}
        <div className="sticky top-0 z-10 flex-shrink-0 flex h-16 bg-white shadow">
          <button
            className="lg:hidden px-4 border-r border-gray-200 text-gray-500 focus:outline-none focus:bg-gray-100"
            onClick={() => setMobileMenuOpen(true)}
          >
            <Menu className="h-6 w-6" />
          </button>
          
          <div className="flex-1 px-4 flex justify-between">
            <div className="flex-1 flex items-center">
              <h1 className="text-xl font-semibold text-gray-900">
                Sheffield Sales Forecast
              </h1>
            </div>
            <div className="ml-4 flex items-center md:ml-6">
              {/* User menu can go here */}
              <div className="bg-gray-100 flex items-center px-3 py-1 rounded-md text-sm font-medium text-gray-700">
                <span>Admin User</span>
              </div>
            </div>
          </div>
        </div>

        {/* Main content area */}
        <main className="flex-1">
          <div className="py-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

export default AppLayout;