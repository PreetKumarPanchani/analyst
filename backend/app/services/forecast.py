import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class ForecastService:
    def __init__(self, data_service):
        """
        Initialize the forecast service with a data service.
        """
        self.data_service = data_service
    
    def forecast_sales(self, company: str, periods: int = 30) -> Dict[str, Any]:
        """
        Forecast sales for the specified company.
        
        Args:
            company: Company to forecast (forge or cpl)
            periods: Number of days to forecast
            
        Returns:
            Dictionary with forecast results and metrics
        """
        # Get the daily sales data
        df = self.data_service.get_daily_data_for_forecast(company)
        
        if df.empty or len(df) < 14:  # Need at least 2 weeks of data
            return {
                "forecast": [],
                "model_metrics": {
                    "mape": 0,
                    "rmse": 0,
                    "mae": 0
                }
            }
        
        # Create and fit Prophet model
        model = Prophet(
            changepoint_prior_scale=0.05,  # More flexible model
            seasonality_prior_scale=10,    # Strong seasonality (weekly patterns)
            seasonality_mode='multiplicative',  # Most retail has multiplicative seasonality
            yearly_seasonality='auto',
            weekly_seasonality=True,
            daily_seasonality=False  # Usually not enough data for daily seasonality
        )
        
        # Add country holidays if we know the location
        model.add_country_holidays(country_name='UK')
        
        # Fit the model
        model.fit(df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Forecast
        forecast = model.predict(future)
        
        # Calculate model metrics using cross-validation
        try:
            # Only do cross validation if we have enough data
            if len(df) >= 60:  # About 2 months of data
                cv_results = cross_validation(
                    model=model,
                    initial=30,  # First 30 days as training
                    period=7,    # Test on 1-week windows
                    horizon=14   # Forecast 2 weeks
                )
                metrics = performance_metrics(cv_results)
                model_metrics = {
                    "mape": float(metrics['mape'].mean()),
                    "rmse": float(metrics['rmse'].mean()),
                    "mae": float(metrics['mae'].mean())
                }
            else:
                # Not enough data for proper cross-validation
                model_metrics = {
                    "mape": float(np.nan),
                    "rmse": float(np.nan),
                    "mae": float(np.nan)
                }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            model_metrics = {
                "mape": float(np.nan),
                "rmse": float(np.nan),
                "mae": float(np.nan)
            }
        
        # Prepare forecast output
        forecast_output = []
        for i, row in forecast.iterrows():
            # Skip historical data, only include future predictions
            if row['ds'] > df['ds'].max():
                forecast_output.append({
                    "date": row['ds'].date(),
                    "sales": float(row['yhat']),
                    "sales_lower": float(row['yhat_lower']),
                    "sales_upper": float(row['yhat_upper'])
                })
        
        return {
            "forecast": forecast_output,
            "model_metrics": model_metrics
        }
    
    def get_forecast_components(self, company: str) -> Dict[str, Any]:
        """
        Get the forecast components (trend, weekly seasonality, yearly seasonality).
        Useful for understanding the patterns in the data.
        """
        # Get the daily sales data
        df = self.data_service.get_daily_data_for_forecast(company)
        
        if df.empty or len(df) < 14:  # Need at least 2 weeks of data
            return {
                "trend": [],
                "weekly": [],
                "yearly": []
            }
        
        # Create and fit Prophet model
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            yearly_seasonality='auto',
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        # Add country holidays
        model.add_country_holidays(country_name='UK')
        
        # Fit the model
        model.fit(df)
        
        # Make future dataframe - cover a full year to see yearly seasonality
        future = model.make_future_dataframe(periods=365)
        
        # Forecast
        forecast = model.predict(future)
        
        # Get the trend component
        trend = forecast[['ds', 'trend']].rename(columns={'ds': 'date', 'trend': 'value'})
        trend['date'] = trend['date'].dt.date
        trend['component'] = 'trend'
        
        # Get the weekly seasonality component
        # Extract only the next 14 days for weekly pattern
        next_two_weeks = forecast[forecast['ds'] <= (df['ds'].max() + timedelta(days=14))]
        weekly = next_two_weeks[['ds', 'weekly']].rename(columns={'ds': 'date', 'weekly': 'value'})
        weekly['date'] = weekly['date'].dt.date
        weekly['component'] = 'weekly'
        
        # Get the yearly seasonality component if available
        if 'yearly' in forecast.columns:
            yearly = forecast[['ds', 'yearly']].rename(columns={'ds': 'date', 'yearly': 'value'})
            yearly['date'] = yearly['date'].dt.date
            yearly['component'] = 'yearly'
        else:
            yearly = pd.DataFrame(columns=['date', 'value', 'component'])
        
        # Combine components
        components = pd.concat([trend, weekly, yearly])
        
        # Convert to list of dictionaries
        components_dict = {
            "trend": trend.to_dict('records'),
            "weekly": weekly.to_dict('records'),
            "yearly": yearly.to_dict('records')
        }
        
        return components_dict


# Test function
def test_forecast_service():
    """Simple test function for the forecast service."""
    from app.services.data_service import DataService
    
    # Initialize services
    data_service = DataService("./test_data")
    forecast_service = ForecastService(data_service)
    
    # Load data
    loaded = data_service.load_data()
    if not loaded:
        print("Data loading failed or no data available")
        return
    
    # Test forecasting
    try:
        # Try to forecast sales for Forge
        forecast_results = forecast_service.forecast_sales("forge", periods=14)
        
        print("\nForecast Results:")
        print(f"Number of forecast points: {len(forecast_results['forecast'])}")
        print(f"Model Metrics: MAPE={forecast_results['model_metrics']['mape']:.2f}, RMSE={forecast_results['model_metrics']['rmse']:.2f}")
        
        if forecast_results['forecast']:
            print("\nSample Forecast:")
            for i, point in enumerate(forecast_results['forecast'][:5]):
                print(f"{point['date']}: £{point['sales']:.2f} (£{point['sales_lower']:.2f} - £{point['sales_upper']:.2f})")
        
        # Try to get forecast components
        components = forecast_service.get_forecast_components("forge")
        print("\nForecast Components:")
        print(f"Trend points: {len(components['trend'])}")
        print(f"Weekly seasonality points: {len(components['weekly'])}")
        print(f"Yearly seasonality points: {len(components['yearly'])}")
        
        print("\nTest completed successfully")
    except Exception as e:
        print(f"Error in forecast test: {e}")

if __name__ == "__main__":
    test_forecast_service()