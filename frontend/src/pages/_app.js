import '../styles/globals.css';
import Head from 'next/head';

function SheffieldSalesForecastApp({ Component, pageProps }) {
  return (
    <>
      <Head>
        <title>Sheffield Sales Forecast</title>
        <meta name="description" content="Sales forecasting application" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Component {...pageProps} />
    </>
  );
}

export default SheffieldSalesForecastApp;