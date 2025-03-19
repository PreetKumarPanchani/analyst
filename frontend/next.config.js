/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // API proxy configuration
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://mm2xymkp2i.eu-west-2.awsapprunner.com/api/v1';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
  // Add output configuration for Amplify
  output: 'standalone',
}

module.exports = nextConfig;