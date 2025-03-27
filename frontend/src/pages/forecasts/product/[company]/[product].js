import { useRouter } from 'next/router';
import { useState, useEffect } from 'react';
import AppLayout from '../../../../components/layout/AppLayout';
import ProductForecastView from '../../../../components/dashboard/ProductForecastView';
import LoadingSpinner from '../../../../components/common/LoadingSpinner';
import ErrorDisplay from '../../../../components/common/ErrorDisplay';
import { fetchApi } from '../../../../utils/apiConfig';

const ProductForecastPage = () => {
  const router = useRouter();
  const { company, product } = router.query;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [validProduct, setValidProduct] = useState(false);

  // Validate company and product params
  useEffect(() => {
    if (!company || !product) return;

    if (!['forge', 'cpl'].includes(company)) {
      router.replace('/dashboard/forge');
      return;
    }

    // Verify product exists for this company
    const validateProduct = async () => {
      try {
        const products = await fetchApi(`/api/v1/sales/products/${company}`);
        const isValid = products.includes(decodeURIComponent(product));
        
        if (!isValid) {
          setError(`Product "${decodeURIComponent(product)}" not found for ${company}`);
        } else {
          setValidProduct(true);
        }
      } catch (err) {
        console.error("Error validating product:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    validateProduct();
  }, [company, product, router]);

  if (!company || !product || loading) {
    return (
      <AppLayout>
        <LoadingSpinner message="Loading product forecast..." />
      </AppLayout>
    );
  }
  
  if (error) {
    return (
      <AppLayout>
        <ErrorDisplay 
          title="Product Not Found" 
          message={error}
          actionText="Go to Dashboard"
          actionHref={`/dashboard/${company}`}
        />
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      {validProduct && (
        <ProductForecastView 
          company={company} 
          initialProduct={decodeURIComponent(product)} 
        />
      )}
    </AppLayout>
  );
};

export default ProductForecastPage;