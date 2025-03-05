// app/layout.tsx
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Bakery Analytics',
  description: 'Analytics dashboard for bakery sales data',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-50 min-h-screen`}>
        <div className="min-h-screen flex flex-col">
          <header className="bg-white shadow-sm">
            <div className="container mx-auto px-4 py-4">
              <div className="flex justify-between items-center">
                <h1 className="text-xl font-bold text-blue-600">Bakery Analytics</h1>
                <nav className="flex space-x-4">
                  <a href="/" className="text-gray-600 hover:text-blue-600">Dashboard</a>
                  <a href="/forecast" className="text-gray-600 hover:text-blue-600">Forecast</a>
                  <a href="/products" className="text-gray-600 hover:text-blue-600">Products</a>
                </nav>
              </div>
            </div>
          </header>
          
          <main className="flex-grow">
            {children}
          </main>
          
          <footer className="bg-white border-t border-gray-200 py-4">
            <div className="container mx-auto px-4">
              <p className="text-center text-gray-500 text-sm">
                &copy; {new Date().getFullYear()} Bakery Analytics. All rights reserved.
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}