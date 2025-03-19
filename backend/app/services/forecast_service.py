# app/services/forecast_service.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.logger import data_logger
from app.services.events_service import EventsService
from app.services.weather_service import WeatherService
from app.data.processor import DataProcessor
#from app.models.prophet_model_enhanced import ProphetModel
from app.models.prophet_model import ProphetModel
from app.core.config import settings
from app.services.s3_service import S3Service
import os


class ForecastService:
    """
    Service for generating sales forecasts for Sheffield companies
    """
    
    def __init__(self, 
                 processed_dir: str = "data/processed", 
                 model_dir: str = "data/models",
                 cache_dir: str = "data/cache",
                 data_dir: str = "data"):
        """Initialize the forecast service"""
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.s3_service = S3Service()
        self.use_s3 = settings.USE_S3_STORAGE
        
        # Create directories if not using S3
        if not self.use_s3:
            os.makedirs(self.processed_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize services
        self.processor = DataProcessor(processed_dir)
        self.prophet_model = ProphetModel(model_dir)
        self.events_service = EventsService(cache_dir)
        self.weather_service = WeatherService(cache_dir=cache_dir)
        
        data_logger.info(f"ForecastService initialized")
    
    def _save_forecast_data(self, model_id: str, data: Dict[str, Any]) -> bool:
        """
        Save forecast data to cache
        
        Args:
            model_id: Model identifier
            data: Forecast data
            
        Returns:
            Success flag
        """
        try:
            if self.use_s3:
                # Save to S3
                s3_key = f"{settings.S3_CACHE_PREFIX}forecast_{model_id}.json"
                
                # Convert data to JSON string
                json_str = json.dumps(data)
                
                # Use put_object directly
                self.s3_service.s3_client.put_object(
                    Bucket=self.s3_service.bucket_name,
                    Key=s3_key,
                    Body=json_str
                )
                
                data_logger.info(f"Saved forecast data to S3: {s3_key}")
            else:
                # Save to local filesystem
                cache_path = os.path.join(self.cache_dir, f"forecast_{model_id}.json")
                
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
                    
                data_logger.info(f"Saved forecast data to: {cache_path}")
                
            return True
            
        except Exception as e:
            data_logger.error(f"Error saving forecast data: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def _load_forecast_data(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load forecast data from cache
        
        Args:
            model_id: Model identifier
            
        Returns:
            Forecast data dictionary (or None if not found)
        """
        try:
            if self.use_s3:
                # Load from S3
                s3_key = f"{settings.S3_CACHE_PREFIX}forecast_{model_id}.json"
                
                if not self.s3_service.file_exists(s3_key):
                    return None
                    
                # Get the JSON from S3
                response = self.s3_service.s3_client.get_object(
                    Bucket=self.s3_service.bucket_name,
                    Key=s3_key
                )
                json_str = response['Body'].read().decode('utf-8')
                
                # Load JSON
                data = json.loads(json_str)
                
                data_logger.info(f"Loaded forecast data from S3: {s3_key}")
                return data
            else:
                # Load from local filesystem
                cache_path = os.path.join(self.cache_dir, f"forecast_{model_id}.json")
                
                if not os.path.exists(cache_path):
                    return None
                    
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    
                data_logger.info(f"Loaded forecast data from: {cache_path}")
                return data
            
        except Exception as e:
            data_logger.error(f"Error loading forecast data: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None
    
    def _save_prophet_data(self, company: str, forecast_type: str, identifier: str, prophet_df: pd.DataFrame) -> bool:
        """
        Save Prophet input data to CSV
        
        Args:
            company: Company name
            forecast_type: Type of forecast (revenue, category, or product)
            identifier: Category or product name (for category/product forecasts)
            prophet_df: Prophet input DataFrame
            
        Returns:
            Success flag
        """
        if not self.use_s3:
            # Create directory path
            dir_path = os.path.join(self.data_dir, company, forecast_type)
            os.makedirs(dir_path, exist_ok=True)
            

        try:

            # Create file name based on forecast type
            if forecast_type == "revenue":
                file_name = f"{company}_prophet_revenue_data.csv"
            elif forecast_type == "category":
                safe_identifier = identifier.replace(" ", "_").replace("/", "_")
                file_name = f"{company}_prophet_{safe_identifier}_category_data.csv"
            elif forecast_type == "product":
                safe_identifier = identifier.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(",", "").replace("-", "_")
                file_name = f"{company}_prophet_{safe_identifier}_product_data.csv"
            else:
                file_name = f"{company}_prophet_{forecast_type}_data.csv"
            
            if self.use_s3:
                # Save to S3
                s3_key = f"{settings.S3_CACHE_PREFIX}{company}/{forecast_type}/{file_name}"
                success = self.s3_service.write_csv_file(prophet_df, s3_key)
                
                if success:
                    data_logger.info(f"Saved Prophet data to S3: {s3_key}")
                    print(f"Prophet dataframe saved to S3: {s3_key}")
                
                return success
            else:
                # Save to local filesystem
                # Create directory path
                dir_path = os.path.join(self.data_dir, company, forecast_type)
                if not self.use_s3: 
                    os.makedirs(dir_path, exist_ok=True)
                
                # Save DataFrame to CSV
                file_path = os.path.join(dir_path, file_name)
                prophet_df.to_csv(file_path, index=False)
                
                data_logger.info(f"Saved Prophet data to: {file_path}")
                print(f"Prophet dataframe saved to: {file_path}")
                
                return True
            
        except Exception as e:
            data_logger.error(f"Error saving Prophet data: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    # Fill the extra ds values in prophet_df, dont fill y values, only ds, it must have the new dates at last and the old dates at first and y values are the same as the old ones
    def fill_extra_ds_values(self, prophet_df: pd.DataFrame, max_date: str, new_max_date: str) -> pd.DataFrame:
        """
        Add future dates to a Prophet DataFrame without filling any values.
        Only adds the 'ds' column entries for future dates.
        
        Args:
            prophet_df: DataFrame with ds (dates) and y (values) columns
            max_date: String representation of the last date in the existing data
            new_max_date: String representation of the last date for the extended range
            
        Returns:
            DataFrame with additional future dates appended
        """
        # Convert string dates to datetime objects
        max_date_dt = pd.to_datetime(max_date)
        new_max_date_dt = pd.to_datetime(new_max_date)
        
        # Create a date range from max_date to new_max_date (starting from day after max_date)
        date_range = pd.date_range(start=max_date_dt + pd.Timedelta(days=1), end=new_max_date_dt)
        
        # Create a new DataFrame with just the date range and no y values
        new_df = pd.DataFrame({'ds': date_range})
        
        # Concatenate original DataFrame with new dates
        extended_df = pd.concat([prophet_df, new_df], ignore_index=True)
        
        return extended_df

    
    def forecast_company_revenue(self, 
                               company: str, 
                               periods: int = 15,
                               include_weather: bool = True,
                               include_events: bool = True,
                               include_time_features: bool = True,
                               force_retrain: bool = False) -> Dict[str, Any]:
        """
        Generate revenue forecast for a company
        
        Args:
            company: Company name
            periods: Number of days to forecast
            include_weather: Whether to include weather data
            include_events: Whether to include events data
            include_time_features: Whether to include time-based features
            force_retrain: Whether to force model retraining
            
        Returns:
            Forecast dictionary
        """
        try:
            # Model identifier
            model_id = f"{company}_revenue"
            if not self.use_s3:
                os.makedirs("debug", exist_ok=True)
            
            # Check if forecast already exists in cache (if not forcing retrain)
            if not force_retrain:
                cached_forecast = self._load_forecast_data(model_id)
                if cached_forecast:
                    data_logger.info(f"Using cached forecast for {model_id}")
                    return cached_forecast
            
            # Load processed data
            processed_data = self.processor.load_processed_data(company)
            
            if not processed_data or "daily_sales" not in processed_data:
                data_logger.error(f"No daily sales data available for company: {company}")
                return {
                    "success": False,
                    "error": f"No data available for company: {company}"
                }
            
            daily_sales = processed_data["daily_sales"]
            
            # Prepare data for Prophet
            prophet_df = self.prophet_model.prepare_data_for_prophet(daily_sales, "total_revenue")
            
            # Get date range for external data
            min_date = daily_sales["date"].min().strftime("%Y-%m-%d")
            max_date = daily_sales["date"].max().strftime("%Y-%m-%d")
            
            # Fill the extra ds values in prophet_df, dont fill y values, only ds
            new_max_date = pd.to_datetime(max_date) + pd.Timedelta(days=periods)
            prophet_df = self.fill_extra_ds_values(prophet_df, max_date, new_max_date)


            # Add weather data if requested
            if include_weather:
                
                data_logger.info(f"Adding weather data to forecast")
                weather_df = self.weather_service.prepare_weather_for_prophet(min_date, max_date, periods)
                
                if not self.use_s3:
                    weather_df.to_csv("debug/weather_df.csv", index=False)
                prophet_df = self.prophet_model.add_weather_features(prophet_df, weather_df)
                if not self.use_s3:
                    prophet_df.to_csv("debug/prophet_df_with_weather.csv", index=False)
            # Add events data if requested
            if include_events:
                data_logger.info(f"Adding events data to forecast")
                events_df = self.events_service.prepare_events_for_prophet(min_date, max_date, periods)
                prophet_df = self.prophet_model.add_events_features(prophet_df, events_df)
                if not self.use_s3:
                    prophet_df.to_csv("debug/prophet_df_with_events.csv", index=False)


            # Add time-based features if requested
            if include_time_features :
                data_logger.info(f"Adding time-based features to forecast")
                prophet_df = self.prophet_model.add_time_features(prophet_df)
                if not self.use_s3:
                    prophet_df.to_csv("debug/prophet_df_with_time_features.csv", index=False)
            
            # Save prophet dataframe for reference and transparency
            self._save_prophet_data(company, "revenue", "revenue", prophet_df)

            try:
                # Generate forecast
                forecast = self.prophet_model.generate_forecast(
                model_id,
                prophet_df,
                periods,
                include_weather,
                include_events,
                include_time_features,
                force_retrain,
                perform_cv=True,
                cv_params=None,
                tune_hyperparams=True,
                param_grid=None

            )
        

            except:
                forecast = self.prophet_model.generate_forecast(
                    model_id,
                    prophet_df,
                    periods,
                    include_weather,
                    include_events,
                    include_time_features,
                    force_retrain,
                )
            
            
            # Add metadata
            forecast["metadata"] = {
                "company": company,
                "target": "revenue",
                "generated_at": datetime.now().isoformat(),
                "periods": periods,
                "include_weather": include_weather,
                "include_events": include_events,
                "include_time_features": include_time_features,
                "force_retrain": force_retrain,
                
                "data_range": {
                    "start": min_date,
                    "end": max_date
                }
            }
            
            # Save forecast data
            self._save_forecast_data(model_id, forecast)
            
            data_logger.info(f"Generated revenue forecast for company: {company}")
            return forecast
            
        except Exception as e:
            data_logger.error(f"Error generating revenue forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def forecast_category_sales(self, 
                              company: str, 
                              category: str,
                              periods: int = 15,
                              include_weather: bool = True,
                              include_events: bool = True,
                              include_time_features: bool = True,
                              force_retrain: bool = False) -> Dict[str, Any]:
        """
        Generate sales forecast for a specific category
        
        Args:
            company: Company name
            category: Product category
            periods: Number of days to forecast
            include_weather: Whether to include weather data
            include_events: Whether to include events data
            include_time_features: Whether to include time-based features
            force_retrain: Whether to force model retraining
            
        Returns:
            Forecast dictionary
        """
        try:
            # Model identifier - make safe for filename
            safe_category = category.replace(" ", "_").replace("/", "_")
            model_id = f"{company}_{safe_category}_category"
            if not self.use_s3:
                os.makedirs("debug", exist_ok=True)
            
            # Check if forecast already exists in cache (if not forcing retrain)
            if not force_retrain:
                cached_forecast = self._load_forecast_data(model_id)
                if cached_forecast:
                    data_logger.info(f"Using cached forecast for {model_id}")
                    return cached_forecast
            
            # Load processed data
            processed_data = self.processor.load_processed_data(company)
            
            if not processed_data or "category_sales" not in processed_data:
                data_logger.error(f"No category sales data available for company: {company}")
                return {
                    "success": False,
                    "error": f"No data available for company: {company}"
                }
            
            category_sales = processed_data["category_sales"]
            
            # Filter for this category
            category_data = category_sales[category_sales["category"] == category]
            
            if len(category_data) == 0:
                data_logger.error(f"No data found for category: {category}")
                return {
                    "success": False,
                    "error": f"No data found for category: {category}"
                }
            
            # Prepare data for Prophet
            prophet_df = self.prophet_model.prepare_data_for_prophet(category_data, "quantity")
            
            # Get date range for external data
            min_date = category_data["date"].min().strftime("%Y-%m-%d")
            max_date = category_data["date"].max().strftime("%Y-%m-%d")
            
            # Fill the extra ds values in prophet_df, dont fill y values, only ds
            new_max_date = pd.to_datetime(max_date) + pd.Timedelta(days=periods)
            prophet_df = self.fill_extra_ds_values(prophet_df, max_date, new_max_date)

            # Add weather data if requested
            if include_weather:
                data_logger.info(f"Adding weather data to forecast")
                weather_df = self.weather_service.prepare_weather_for_prophet(min_date, max_date, periods)
                prophet_df = self.prophet_model.add_weather_features(prophet_df, weather_df)
                
                # Save prophet data with weather for debugging
                if not self.use_s3:
                    prophet_df.to_csv("debug/prophet_df_with_weather_category.csv", index=False)

            # Add events data if requested
            if include_events:
                data_logger.info(f"Adding events data to forecast")
                events_df = self.events_service.prepare_events_for_prophet(min_date, max_date, periods)
                prophet_df = self.prophet_model.add_events_features(prophet_df, events_df)

                # Save prophet data with events for debugging
                if not self.use_s3:
                    prophet_df.to_csv("debug/prophet_df_with_events_category.csv", index=False)

            # Add time-based features if requested
            if include_time_features:
                data_logger.info(f"Adding time-based features to forecast")
                prophet_df = self.prophet_model.add_time_features(prophet_df)

                # Save prophet data with time features for debugging
                if not self.use_s3:
                    prophet_df.to_csv("debug/prophet_df_with_time_features_category.csv", index=False)

            # Save prophet dataframe for reference and transparency
            self._save_prophet_data(company, "category", category, prophet_df)
            
            try:
                # Generate forecast
                forecast = self.prophet_model.generate_forecast(
                model_id,
                prophet_df,
                periods,
                include_weather,
                include_events,
                include_time_features,
                force_retrain,
                perform_cv=True,
                cv_params=None,
                tune_hyperparams=True,
                param_grid=None,

            )
            except:
                forecast = self.prophet_model.generate_forecast(
                    model_id,
                    prophet_df,
                    periods,
                    include_weather,
                    include_events,
                    include_time_features,
                    force_retrain,
                )
            

            # Add metadata
            forecast["metadata"] = {
                "company": company,
                "category": category,
                "target": "quantity",
                "generated_at": datetime.now().isoformat(),
                "periods": periods,
                "include_weather": include_weather,
                "include_events": include_events,
                "include_time_features": include_time_features,
                "force_retrain": force_retrain,
                "data_range": {
                    "start": min_date,
                    "end": max_date
                }
            }
            
            # Save forecast data
            self._save_forecast_data(model_id, forecast)
            
            data_logger.info(f"Generated sales forecast for company: {company}, category: {category}")
            return forecast
            
        except Exception as e:
            data_logger.error(f"Error generating category sales forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def forecast_product_sales(self, 
                             company: str, 
                             product: str,
                             periods: int = 15,
                             include_weather: bool = True,
                             include_events: bool = True,
                             include_time_features: bool = True,
                             force_retrain: bool = False) -> Dict[str, Any]:
        """
        Generate sales forecast for a specific product
        
        Args:
            company: Company name
            product: Product name
            periods: Number of days to forecast
            include_weather: Whether to include weather data
            include_events: Whether to include events data
            include_time_features: Whether to include time-based features
            force_retrain: Whether to force model retraining
            
        Returns:
            Forecast dictionary
        """
        try:
            # Model identifier - replace spaces and special chars
            safe_product = product.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("-", "_").replace("/", "_")
            model_id = f"{company}_{safe_product}_product"
            if not self.use_s3:
                os.makedirs("debug", exist_ok=True)
            

            # Check if forecast already exists in cache (if not forcing retrain)
            if not force_retrain:
                cached_forecast = self._load_forecast_data(model_id)
                if cached_forecast:
                    data_logger.info(f"Using cached forecast for {model_id}")
                    return cached_forecast
            
            # Load processed data
            processed_data = self.processor.load_processed_data(company)
            
            if not processed_data or "product_sales" not in processed_data:
                data_logger.error(f"No product sales data available for company: {company}")
                return {
                    "success": False,
                    "error": f"No data available for company: {company}"
                }
            
            product_sales = processed_data["product_sales"]
            
            # Filter for this product
            product_data = product_sales[product_sales["product"] == product]
            
            if len(product_data) == 0:
                data_logger.error(f"No data found for product: {product}")
                return {
                    "success": False,
                    "error": f"No data found for product: {product}"
                }
            
            # Prepare data for Prophet
            prophet_df = self.prophet_model.prepare_data_for_prophet(product_data, "quantity")
            
            # Get date range for external data
            min_date = product_data["date"].min().strftime("%Y-%m-%d")
            max_date = product_data["date"].max().strftime("%Y-%m-%d")
            

            # Fill the extra ds values in prophet_df, dont fill y values, only ds
            new_max_date = pd.to_datetime(max_date) + pd.Timedelta(days=periods)
            prophet_df = self.fill_extra_ds_values(prophet_df, max_date, new_max_date)


            # Add weather data if requested
            if include_weather:
                data_logger.info(f"Adding weather data to forecast")
                weather_df = self.weather_service.prepare_weather_for_prophet(min_date, max_date, periods)
                prophet_df = self.prophet_model.add_weather_features(prophet_df, weather_df)
                
                # Save prophet data with weather for debugging
                if not self.use_s3:
                    prophet_df.to_csv("debug/prophet_df_with_weather_product.csv", index=False)

            # Add events data if requested
            if include_events:
                data_logger.info(f"Adding events data to forecast")
                events_df = self.events_service.prepare_events_for_prophet(min_date, max_date, periods)
                prophet_df = self.prophet_model.add_events_features(prophet_df, events_df)
                
                # Save prophet data with events for debugging
                if not self.use_s3:
                    prophet_df.to_csv("debug/prophet_df_with_events_product.csv", index=False)

            # Add time-based features if requested
            if include_time_features:
                data_logger.info(f"Adding time-based features to forecast")
                prophet_df = self.prophet_model.add_time_features(prophet_df)
            
            # Save prophet data with time features for debugging
                if not self.use_s3:
                    prophet_df.to_csv("debug/prophet_df_with_time_features_product.csv", index=False)
            
            # Save prophet dataframe for reference and transparency
            self._save_prophet_data(company, "product", product, prophet_df)

            try:
                # Generate forecast
                forecast = self.prophet_model.generate_forecast(
                    model_id,
                    prophet_df,
                    periods,
                    include_weather,
                    include_events,
                    include_time_features,
                    force_retrain,
                    perform_cv=True,
                    cv_params=None,
                    tune_hyperparams=True,
                    param_grid=None,

                )
            except:
                forecast = self.prophet_model.generate_forecast(
                    model_id,
                    prophet_df,
                    periods,
                    include_weather,
                    include_events,
                    include_time_features,
                    force_retrain,
                )

            
            # Add metadata
            forecast["metadata"] = {
                "company": company,
                "product": product,
                "target": "quantity",
                "generated_at": datetime.now().isoformat(),
                "periods": periods,
                "include_weather": include_weather,
                "include_events": include_events,
                "include_time_features": include_time_features,
                "force_retrain": force_retrain,
                "data_range": {
                    "start": min_date,
                    "end": max_date
                }
            }
            
            # Save forecast data
            self._save_forecast_data(model_id, forecast)
            
            data_logger.info(f"Generated sales forecast for company: {company}, product: {product}")
            return forecast
            
        except Exception as e:
            data_logger.error(f"Error generating product sales forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_top_products(self, 
                        company: str, 
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top products by sales quantity
        
        Args:
            company: Company name
            limit: Number of products to return
            
        Returns:
            List of top products with sales data
        """
        try:
            # Load processed data
            processed_data = self.processor.load_processed_data(company)
            
            if not processed_data or "product_sales" not in processed_data:
                data_logger.error(f"No product sales data available for company: {company}")
                return []
            
            product_sales = processed_data["product_sales"]
            
            # Group by product and sum quantity
            product_totals = product_sales.groupby("product").agg({
                "quantity": "sum",
                "revenue": "sum"
            }).reset_index()
            
            # Sort by quantity and get top products
            top_products = product_totals.sort_values("quantity", ascending=False).head(limit)
            
            # Convert to list of dictionaries
            result = []
            for _, row in top_products.iterrows():
                result.append({
                    "product": row["product"],
                    "total_quantity": int(row["quantity"]),
                    "total_revenue": float(row["revenue"])
                })
            
            data_logger.info(f"Retrieved top {len(result)} products for company: {company}")
            return result
            
        except Exception as e:
            data_logger.error(f"Error getting top products: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []
    
    def get_categories(self, company: str) -> List[str]:
        """
        Get list of available product categories
        
        Args:
            company: Company name
            
        Returns:
            List of category names
        """
        try:
            # Load processed data
            processed_data = self.processor.load_processed_data(company)
            
            if not processed_data or "category_sales" not in processed_data:
                data_logger.error(f"No category sales data available for company: {company}")
                return []
            
            category_sales = processed_data["category_sales"]
            
            # Get unique categories
            categories = category_sales["category"].unique().tolist()
            
            data_logger.info(f"Retrieved {len(categories)} categories for company: {company}")
            return categories
            
        except Exception as e:
            data_logger.error(f"Error getting categories: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []
    

    def get_products(self, company: str, category: Optional[str] = None) -> List[str]:
        """
        Get list of available products
        
        Args:
            company: Company name
            category: Filter by category (optional)
            
        Returns:
            List of product names
        """
        try:
            # Load processed data
            processed_data = self.processor.load_processed_data(company)

            print(processed_data)
            
            if not processed_data or "product_sales" not in processed_data:
                data_logger.error(f"No product sales data available for company: {company}")
                return []
            
            product_sales = processed_data["product_sales"]
            
            # Filter by category if specified
            if category:
                # Try to load the product-category mapping file
                mapping_path = os.path.join(self.processed_dir, company, "product_category_map.csv")
                
                if os.path.exists(mapping_path):
                    # Load the mapping and filter by category
                    product_category_map = pd.read_csv(mapping_path)
                    category_products = product_category_map[product_category_map["category"].str.lower() == category.lower()]["product"].tolist()
                    
                    # Get products in this category that also exist in product_sales
                    product_list = product_sales[product_sales["product"].isin(category_products)]["product"].unique().tolist()
                else:
                    # If mapping doesn't exist yet, create it on the fly
                    data_logger.warning(f"Product-category mapping not found, creating on the fly")
                    from app.data.loader import DataLoader
                    
                    loader = DataLoader()
                    raw_data = loader.load_company_data(company)
                    
                    if raw_data and "sales_items" in raw_data:
                        # Generate the mapping and save it for future use
                        product_category_map = self.processor.generate_product_category_mapping(company, raw_data["sales_items"])
                        if not product_category_map.empty:
                            category_products = product_category_map[product_category_map["category"].str.lower() == category.lower()]["product"].tolist()
                            product_list = product_sales[product_sales["product"].isin(category_products)]["product"].unique().tolist()
                        else:
                            product_list = []
                    else:
                        data_logger.error(f"Could not load raw data to create product-category mapping")
                        product_list = []
            else:
                # Get all products if no category filter
                product_list = product_sales["product"].unique().tolist()
            
            data_logger.info(f"Retrieved {len(product_list)} products for company: {company}, category: {category if category else 'all'}")
            return product_list
            
        except Exception as e:
            data_logger.error(f"Error getting products: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []
    

def test_forecast_service():
    """Test the ForecastService functionality"""
    from app.data.loader import DataLoader
    
    # Initialize services
    loader = DataLoader()
    service = ForecastService()
    
    # Test for Forge
    company = "forge"
    
    # Load raw data if processed data doesn't exist
    processed_data = service.processor.load_processed_data(company)
    if not processed_data:
        print("Processing raw data...")
        raw_data = loader.load_company_data(company)
        processed_data = service.processor.process_company_data(company, raw_data)
    
    # Generate revenue forecast
    print("\n=== Testing Revenue Forecast ===")
    revenue_forecast = service.forecast_company_revenue(
        company, 
        periods=5,
        include_weather=True,
        include_events=True,
        include_time_features=True
    )
    
    print(f"Revenue forecast success: {revenue_forecast['success']}")
    if revenue_forecast["success"]:
        print(f"Forecast periods: {len(revenue_forecast['dates']) - len(revenue_forecast['actuals'])}")
        if revenue_forecast['metrics']['mape'] is not None:
            print(f"MAPE: {revenue_forecast['metrics']['mape']:.2f}%")
    
    # Test force retrain
    print("\n=== Testing Force Retrain ===")
    retrained_forecast = service.forecast_company_revenue(
        company, 
        periods=5,
        include_weather=True,
        include_events=True,
        include_time_features=True,
        force_retrain=True
    )
    print(f"Forced retrain (different timestamp): {retrained_forecast['metadata']['generated_at'] != revenue_forecast['metadata']['generated_at']}")
    
    # Get top products
    print("\n=== Testing Top Products ===")
    top_products = service.get_top_products(company, limit=2)
    
    for product in top_products:
        print(f"{product['product']}: {product['total_quantity']} units, £{product['total_revenue']:.2f}")
        
        # Generate product forecast
        print(f"Generating forecast for {product['product']}...")
        product_forecast = service.forecast_product_sales(
            company,
            product['product'],
            periods=5,
            include_weather=True,
            include_events=True,
            include_time_features=True
        )
        
        if product_forecast["success"]:
            print(f"  Forecast success for {product['product']}: {product_forecast['success']}")
            if product_forecast['metrics']['mape'] is not None:
                print(f"  MAPE: {product_forecast['metrics']['mape']:.2f}%")
    
    # Get categories
    print("\n=== Testing Categories ===")
    categories = service.get_categories(company)
    print(f"First 2 categories: {categories[:2]}...")

    # Generate category forecast
    print("\n=== Testing Category Forecast ===")
    if categories:
        category = categories[0]
        print(f"Generating forecast for category: {category}")
        
        category_forecast = service.forecast_category_sales(
            company,
            category,
            periods=5,
            include_weather=True,
            include_events=True,
            include_time_features=True
        )

        if category_forecast["success"]:
            print(f"Forecast success for {category}: {category_forecast['success']}")
            if category_forecast['metrics']['mape'] is not None:
                print(f"MAPE: {category_forecast['metrics']['mape']:.2f}%")

    return "Forecast service test completed successfully"

if __name__ == "__main__":
    test_forecast_service()