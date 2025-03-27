/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // Static export configuration for Next.js 13+
  output: 'export',
  
  // Required for static export with images
  images: {
    unoptimized: true,
  },
  
  /*
  // API proxy configuration (will work during development but not in static export)
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://mm2xymkp2i.eu-west-2.awsapprunner.com/api/v1';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/:path*`,
      },
    ];
  }
  */
  // Environment variable configuration
  env: {
    //NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://mm2xymkp2i.eu-west-2.awsapprunner.com/api/v1',
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001/api/v1'
  }
}

module.exports = nextConfig;