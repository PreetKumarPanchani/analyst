# app/models/prophet_model.py
import os
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import traceback
import time


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.logger import data_logger
from app.core.config import settings
from app.services.s3_service import S3Service

class ProphetModel:
    """
    Prophet model for multivariate time series forecasting with external regressors
    """
    
    def __init__(self, model_dir: str = "data/models"):
        """Initialize the Prophet model"""
        self.model_dir = model_dir
        self.s3_service = S3Service()
        self.use_s3 = settings.USE_S3_STORAGE
        
        if not self.use_s3:
            os.makedirs(self.model_dir, exist_ok=True)
            
        data_logger.info(f"ProphetModel initialized with model directory: {model_dir}")
    
    def prepare_data_for_prophet(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Prepare data for Prophet model
        
        Args:
            df: Input DataFrame with 'date' column
            target_col: Target column for forecasting
            
        Returns:
            DataFrame formatted for Prophet
        """
        try:
            # Create Prophet format dataframe
            prophet_df = df[["date", target_col]].copy()
            
            # Rename columns to Prophet's required format
            prophet_df.columns = ["ds", "y"]
            
            # Sort by date
            prophet_df = prophet_df.sort_values("ds")
            
            # Ensure date column is datetime
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
            
            # Add time-based features if they exist in the original dataframe
            for feature in ['dayofweek', 'is_weekend', 'month', 'year']:
                if feature in df.columns:

                    prophet_df[feature] = df[feature].values
            
            data_logger.info(f"Prepared data for Prophet with {len(prophet_df)} rows")
            return prophet_df
            
        except Exception as e:
            data_logger.error(f"Error preparing data for Prophet: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def add_events_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add events features to the input DataFrame
        
        Args:
            df: Input DataFrame
            events_df: Events DataFrame
            
        Returns:
            DataFrame with added events features
        """
        try:
            if events_df.empty:
                data_logger.warning("No events data provided")
                return df
                
            # Copy input DataFrame
            result_df = df.copy()
            
            # Merge with events data
            result_df = pd.merge(
                result_df,
                events_df[["ds", "event", "holiday", "festival",]],
                on="ds",
                how="left"
            )
            
            # Fill NaN values
            result_df["event"] = result_df["event"].fillna(0)
            result_df["holiday"] = result_df["holiday"].fillna(0)
            result_df["festival"] = result_df["festival"].fillna(0)
            
            data_logger.info(f"Added events features: {events_df['event'].sum()} events")
            return result_df
            
        except Exception as e:
            data_logger.error(f"Error adding events features: {str(e)}")
            data_logger.error(traceback.format_exc())
            return df
    

    '''
    def add_weather_features(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced weather features to the input DataFrame
        
        Args:
            df: Input DataFrame
            weather_df: Weather DataFrame with enhanced features
            
        Returns:
            DataFrame with added weather features
        """
        try:
            if weather_df.empty:
                data_logger.warning("No weather data provided")
                return df
                
            # Copy input DataFrame
            result_df = df.copy()
            
            # Get all available weather columns, excluding 'ds'
            weather_columns = [col for col in weather_df.columns if col != 'ds']
            
            # Merge with weather data on the date field
            result_df = pd.merge(
                result_df,
                weather_df[['ds'] + weather_columns],
                on='ds',
                how='left'
            )
            
            # Log which features were added
            data_logger.info(f"Added weather features: {', '.join(weather_columns)}")
            
            # Fill NaN values with appropriate defaults
            for col in weather_columns:
                if col in result_df.columns:
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        if col in ['is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy'] or col.startswith('is_'):
                            # Binary columns get 0
                            result_df[col] = result_df[col].fillna(0)
                        elif 'precipitation' in col or 'rain' in col or 'snow' in col:
                            # Precipitation values get 0
                            result_df[col] = result_df[col].fillna(0)
                        else:
                            # Other numeric columns get filled with their mean
                            mean_value = result_df[col].mean()
                            result_df[col] = result_df[col].fillna(mean_value)
            
            return result_df
            
        except Exception as e:
            data_logger.error(f"Error adding weather features: {str(e)}")
            data_logger.error(traceback.format_exc())
            return df
    '''

    def add_weather_features(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced weather features to the input DataFrame
        
        Args:
            df: Input DataFrame
            weather_df: Weather DataFrame with enhanced features (including future forecasts)
            
        Returns:
            DataFrame with added weather features
        """
        try:
            if weather_df.empty:
                data_logger.warning("No weather data provided")
                return df
                
            # Copy input DataFrame
            result_df = df.copy()
            
            # Get all available weather columns, excluding 'ds'
            weather_columns = [col for col in weather_df.columns if col != 'ds']
            
            # Verify that weather_df contains all dates from df, including future dates
            missing_dates = set(pd.to_datetime(result_df['ds'])) - set(pd.to_datetime(weather_df['ds']))
            if missing_dates:
                data_logger.warning(f"Weather data missing for {len(missing_dates)} dates. First few: {list(missing_dates)[:5]}")
            
            # Ensure both dataframes have datetime type for 'ds' to ensure proper merging
            result_df['ds'] = pd.to_datetime(result_df['ds'])
            weather_df['ds'] = pd.to_datetime(weather_df['ds'])
            
            # Merge with weather data on the date field - this will bring in the weather data for all dates
            # that exist in weather_df, including future dates
            result_df = pd.merge(
                result_df,
                weather_df[['ds'] + weather_columns],
                on='ds',
                how='left'
            )
            
            # Log which features were added
            data_logger.info(f"Added weather features: {', '.join(weather_columns)}")
            
            # Fill NaN values only for dates that don't have weather data
            # This should only happen if weather_df is missing some dates that are in df
            for col in weather_columns:
                if col in result_df.columns and result_df[col].isnull().any():
                    missing_count = result_df[col].isnull().sum()
                    data_logger.warning(f"Missing {missing_count} values for {col}, filling with defaults")
                    
                    if col in ['is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy'] or col.startswith('is_'):
                        # Binary columns get 0
                        result_df[col] = result_df[col].fillna(0)
                    elif 'precipitation' in col or 'rain' in col or 'snow' in col:
                        # Precipitation values get 0
                        result_df[col] = result_df[col].fillna(0)
                    else:
                        # Other numeric columns get filled with their mean
                        # Calculate mean only from historical values, not including future dates
                        historical_mean = result_df.loc[result_df['ds'] <= pd.Timestamp.now(), col].mean()
                        result_df[col] = result_df[col].fillna(historical_mean)
            
            return result_df
            
        except Exception as e:
            data_logger.error(f"Error adding weather features: {str(e)}")
            data_logger.error(traceback.format_exc())
            return df
        


    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add or ensure time-based features to the input DataFrame
        
        Args:
            df: Input DataFrame with 'ds' column
            
        Returns:
            DataFrame with added time features
        """
        try:
            # Copy input DataFrame
            result_df = df.copy()
            
            # Add time-based features if not already present or missing values
            if 'dayofweek' not in result_df.columns or result_df['dayofweek'].isnull().any():
                result_df['dayofweek'] = result_df['ds'].dt.dayofweek
            if 'month' not in result_df.columns or result_df['month'].isnull().any():
                result_df['month'] = result_df['ds'].dt.month
            if 'year' not in result_df.columns or result_df['year'].isnull().any():
                result_df['year'] = result_df['ds'].dt.year
            if 'is_weekend' not in result_df.columns or result_df['is_weekend'].isnull().any():
                result_df['is_weekend'] = (result_df['ds'].dt.dayofweek >= 5).astype(int)
            
            data_logger.info(f"Added time-based features")
            return result_df
            
        except Exception as e:
            data_logger.error(f"Error adding time features: {str(e)}")
            data_logger.error(traceback.format_exc())
            return df
    


    '''
    def train_model(self, 
                    df: pd.DataFrame, 
                    model_id: str,
                    periods: int = 15,
                    include_weather: bool = True,
                    include_events: bool = True,
                    include_time_features: bool = True) -> Tuple[Prophet, pd.DataFrame]:
        """
        Train a Prophet model with enhanced weather regressors
        
        Args:
            df: Input DataFrame in Prophet format (with ds, y)
            model_id: Unique identifier for the model
            periods: Number of periods to forecast
            include_weather: Whether to include weather as regressors
            include_events: Whether to include events as regressors
            include_time_features: Whether to include time-based features as regressors
            
        Returns:
            Tuple of (Prophet model, forecast DataFrame)
        """

        ## Based on y values, select the train df i.e if y not present

        
        # Create future df from df , remove y and rest is fine
        

        try:
            data_logger.info(f"Training Prophet model: {model_id}")
            
            # Initialize Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            # Add enhanced weather regressors if present
            if include_weather:

                # Core weather metrics (numerical variables)
                weather_numerical = [
                    'temperature', 'min_temp', 'max_temp', 'min_feels_like', 'max_feels_like',
                    'rain', 'snowfall', 'precipitation', 'wind_speed', 'radiation', 
                    'temp_delta', 'feels_like', 'precipitation_hours', 'precipitation_intensity',
                ]
                
                # Weather category flags (binary variables)
                weather_categorical = [
                    'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy'
                ]
                
                # Add each weather feature that exists in the dataframe
                for feature in weather_numerical + weather_categorical:
                    if feature in df.columns:
                        data_logger.info(f"Adding weather regressor: {feature}")
                        # Standardize numeric features (except binary ones)
                        standardize = feature not in weather_categorical
                        model.add_regressor(feature, standardize=standardize)
            
            # Add event regressors if present
            if include_events:
                event_features = ['event', 'holiday', 'festival']
                for feature in event_features:
                    if feature in df.columns:
                        data_logger.info(f"Adding event regressor: {feature}")
                        model.add_regressor(feature, standardize=False)  # Don't standardize binary features

            # Add time-based features as regressors if requested 
            if include_time_features:
                # Ensure time features exist
                df = self.add_time_features(df)
                
            # Add time-based regressors
            model.add_regressor('dayofweek', standardize=False)
            model.add_regressor('is_weekend', standardize=False)
            model.add_regressor('month', standardize=False)
            model.add_regressor('year', standardize=False)

            
            # Save the input dataframe to a csv file
            df.to_csv(os.path.join(self.model_dir, f"{model_id}_model_input_dataframe.csv"), index=False)

            # Fit the model
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)


            # Add time-based features to future
            if include_time_features:
                future = self.add_time_features(future)
            
            # Add regressor values to future
            for col in df.columns:
                if col not in ['ds', 'y', 'dayofweek', 'is_weekend', 'month', 'year']:
                    # Only add if it's needed for a regressor
                    if col in model.extra_regressors:
                        # Initialize with appropriate default
                        if col in weather_categorical or col in event_features:
                            # For binary features, use 0
                            future[col] = 0
                        elif 'precipitation' in col or 'rain' in col or 'snow' in col:
                            # For precipitation features, use 0
                            future[col] = 0
                        else:
                            # For other features, use the mean from historical data
                            future[col] = df[col].mean()
                        
                        # Copy known values from historical data to future dataframe
                        for idx, row in df.iterrows():
                            mask = future['ds'] == row['ds']
                            if any(mask):
                                future.loc[mask, col] = row[col]
            
            # reorder the columns to be: ds dayofweek	is_weekend	month	year	temperature	min_temp	max_temp	min_feels_like	max_feels_like	rain	snowfall	precipitation	wind_speed	     radiation  	is_rainy	is_snowy	is_sunny	is_cloudy	temp_delta	feels_like	  precipitation_hours	    precipitation_intensity	  event	   holiday	festival
            future = future[['ds', 'dayofweek', 'is_weekend', 'month', 'year', 'temperature', 'min_temp', 'max_temp', 'min_feels_like', 'max_feels_like', 'rain', 'snowfall', 'precipitation', 'wind_speed', 'radiation', 'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy', 'temp_delta', 'feels_like', 'precipitation_hours', 'precipitation_intensity', 'event', 'holiday', 'festival']]

            
            # Save the forecast dataframe to a csv file
            future.to_csv(os.path.join(self.model_dir, f"new_model_{model_id}_future_dataframe.csv"), index=False)
            
            # Make forecast
            forecast = model.predict(future)
            
            # Save model
            self._save_model(model, model_id)
            
            data_logger.info(f"Prophet model trained successfully: {model_id}")
            return model, forecast
            
        except Exception as e:
            data_logger.error(f"Error training Prophet model: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None, pd.DataFrame()
    
    '''


    def train_model(self, 
                    df: pd.DataFrame, 
                    model_id: str,
                    periods: int = 15,
                    include_weather: bool = True,
                    include_events: bool = True,
                    include_time_features: bool = True) -> Tuple[Prophet, pd.DataFrame]:
        """
        Train a Prophet model with enhanced weather regressors
        
        Args:
            df: Input DataFrame containing both historical data (with 'y') and future data (without 'y')
            model_id: Unique identifier for the model
            periods: Number of periods to forecast (not used if future data is already in df)
            include_weather: Whether to include weather as regressors
            include_events: Whether to include events as regressors
            include_time_features: Whether to include time-based features as regressors
            
        Returns:
            Tuple of (Prophet model, forecast DataFrame)
        """
        try:
            data_logger.info(f"Training Prophet model: {model_id}")
            
            # Split dataframe into historical (training) and future data
            historical_df = df.dropna(subset=['y']).copy()
            future_df = df.copy()  # Keep all rows for prediction
            
            data_logger.info(f"Training on {len(historical_df)} historical rows and predicting for {len(future_df)} total rows")
            
            # Initialize Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            # Add enhanced weather regressors if present
            if include_weather:
                # Core weather metrics (numerical variables)
                weather_numerical = [
                    'temperature', 'min_temp', 'max_temp', 'min_feels_like', 'max_feels_like',
                    'rain', 'snowfall', 'precipitation', 'wind_speed', 'radiation', 
                    'temp_delta', 'feels_like', 'precipitation_hours', 'precipitation_intensity',
                ]
                
                # Weather category flags (binary variables)
                weather_categorical = [
                    'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy'
                ]
                
                # Add each weather feature that exists in the dataframe
                for feature in weather_numerical + weather_categorical:
                    if feature in df.columns:
                        data_logger.info(f"Adding weather regressor: {feature}")
                        # Standardize numeric features (except binary ones)
                        standardize = feature not in weather_categorical
                        model.add_regressor(feature, standardize=standardize)
            
            # Add event regressors if present
            if include_events:
                event_features = ['event', 'holiday', 'festival']
                for feature in event_features:
                    if feature in df.columns:
                        data_logger.info(f"Adding event regressor: {feature}")
                        model.add_regressor(feature, standardize=False)  # Don't standardize binary features

            # Add time-based features as regressors if requested 
            if include_time_features:
                # Ensure time features exist in both historical and future dataframes
                historical_df = self.add_time_features(historical_df)
                future_df = self.add_time_features(future_df)
                
            # Add time-based regressors
            model.add_regressor('dayofweek', standardize=False)
            model.add_regressor('is_weekend', standardize=False)
            model.add_regressor('month', standardize=False)
            model.add_regressor('year', standardize=False)

            
            # Save the input training dataframe to a csv file
            historical_df.to_csv(os.path.join(self.model_dir, f"{model_id}_model_input_training_dataframe.csv"), index=False)

            # Fit the model on historical data only
            model.fit(historical_df)
            
            # Save the full future dataframe to a csv file
            future_df.to_csv(os.path.join(self.model_dir, f"{model_id}_future_dataframe.csv"), index=False)

            # Make forecast on the entire dataset
            forecast = model.predict(future_df)
            
            # Save model
            self._save_model(model, model_id)
            
            data_logger.info(f"Prophet model trained successfully: {model_id}")
            return model, forecast
            
        except Exception as e:
            data_logger.error(f"Error training Prophet model: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None, pd.DataFrame()
        
        

    '''
    def generate_forecast(self, 
                        model_id: str, 
                        df: pd.DataFrame, 
                        periods: int = 15,
                        include_weather: bool = True,
                        include_events: bool = True,
                        include_time_features: bool = True,
                        force_retrain: bool = False) -> Dict[str, Any]:
        """
        Generate forecast using a saved model or train a new one
        
        Args:
            model_id: Unique identifier for the model
            df: Input DataFrame in Prophet format
            periods: Number of periods to forecast
            include_weather: Whether to include weather as regressors
            include_events: Whether to include events as regressors
            include_time_features: Whether to include time-based features
            force_retrain: Whether to force model retraining
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Try to load existing model (unless force_retrain is True)
            model = None if force_retrain else self.load_model(model_id)
            
            # Train new model if not found or force_retrain is True
            if model is None or force_retrain:
                data_logger.info(f"Training new model (model not found or force_retrain={force_retrain})")
                model, forecast = self.train_model(
                    df, 
                    model_id, 
                    periods, 
                    include_weather, 
                    include_events,
                    include_time_features
                )
            else:
                data_logger.info(f"Using existing model: {model_id}")
                # Create future dataframe
                future = model.make_future_dataframe(periods=periods) 
                
                # Add time-based features to future
                if include_time_features:
                    future = self.add_time_features(future)
                
                # Get list of extra regressors needed by the model
                regressor_names = list(model.extra_regressors.keys())
                
                # Add regressor values to future
                for col in regressor_names:
                    if col not in ['dayofweek', 'is_weekend', 'month', 'year']:
                        # Check if the regressor exists in the input data
                        if col in df.columns:
                            # Initialize with appropriate default based on column type
                            if col.startswith('is_') or col in ['event', 'holiday', 'festival']:
                                # For binary features, use 0
                                future[col] = 0
                            elif 'precipitation' in col or 'rain' in col or 'snow' in col:
                                # For precipitation features, use 0
                                future[col] = 0
                            else:
                                # For other features, use the mean from historical data
                                future[col] = df[col].mean()
                            
                            # Copy known values from historical data to future dataframe
                            for idx, row in df.iterrows():
                                mask = future['ds'] == row['ds']
                                if any(mask):
                                    future.loc[mask, col] = row[col]
                        else:
                            data_logger.warning(f"Regressor {col} not found in input data")
                            # Initialize with zeros
                            future[col] = 0
                

                # reorder the columns to be: ds dayofweek	is_weekend	month	year	temperature	min_temp	max_temp	min_feels_like	max_feels_like	rain	snowfall	precipitation	wind_speed	     radiation  	is_rainy	is_snowy	is_sunny	is_cloudy	temp_delta	feels_like	  precipitation_hours	    precipitation_intensity	  event	   holiday	festival
                future = future[['ds', 'dayofweek', 'is_weekend', 'month', 'year', 'temperature', 'min_temp', 'max_temp', 'min_feels_like', 'max_feels_like', 'rain', 'snowfall', 'precipitation', 'wind_speed', 'radiation', 'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy', 'temp_delta', 'feels_like', 'precipitation_hours', 'precipitation_intensity', 'event', 'holiday', 'festival']]


                # Save the future dataframe to a csv file
                future.to_csv(os.path.join(self.model_dir, f"loaded_model_{model_id}_future_dataframe.csv"), index=False)


                # Make forecast
                forecast = model.predict(future)
            
            if model is None:
                return {
                    "success": False,
                    "error": "Failed to create or load model"
                }
                
            # Merge with original data
            result = pd.merge(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                df[['ds', 'y']],
                on='ds',
                how='left'
            )
            
            # Calculate accuracy metrics for historical period
            historical = result.dropna(subset=['y'])
            
            if len(historical) > 0:
                mape = np.mean(np.abs((historical['y'] - historical['yhat']) / historical['y'])) * 100
                rmse = np.sqrt(np.mean((historical['y'] - historical['yhat']) ** 2))
            else:
                mape = None
                rmse = None
            
            # Split into historical and forecast periods
            historical_data = result[result['ds'] <= df['ds'].max()]
            forecast_data = result[result['ds'] > df['ds'].max()]
            
            # Convert to lists for JSON serialization
            dates = result['ds'].dt.strftime('%Y-%m-%d').tolist()
            actuals = result['y'].tolist()
            predictions = result['yhat'].tolist()
            lower_bound = result['yhat_lower'].tolist()
            upper_bound = result['yhat_upper'].tolist()
            
            # Extract model components
            components = {
                "trend": forecast['trend'].tolist(),
                "weekly": forecast['weekly'].tolist() if 'weekly' in forecast.columns else None,
                "yearly": forecast['yearly'].tolist() if 'yearly' in forecast.columns else None
            }
            
            # Add weather components if available
            weather_components = {}
            for col in forecast.columns:
                if col.startswith('extra_regressors_'):
                    # Extract regressor name from column name
                    regressor_name = col.replace('extra_regressors_', '')
                    if regressor_name in df.columns:
                        weather_components[regressor_name] = forecast[col].tolist()
            
            # Add weather components to the response if any were added
            if weather_components:
                components["weather"] = weather_components
            
            # Create forecast response
            response = {
                "success": True,
                "model_id": model_id,
                "dates": dates,
                "actuals": actuals,
                "predictions": predictions,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "metrics": {
                    "mape": mape,
                    "rmse": rmse
                },
                "components": components
            }
            
            data_logger.info(f"Generated forecast for model: {model_id}")
            return response
            
        except Exception as e:
            data_logger.error(f"Error generating forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
        
    '''




    def generate_forecast(self, 
                        model_id: str, 
                        df: pd.DataFrame, 
                        periods: int = 15,
                        include_weather: bool = True,
                        include_events: bool = True,
                        include_time_features: bool = True,
                        force_retrain: bool = False) -> Dict[str, Any]:
        """
        Generate forecast using a saved model or train a new one
        
        Args:
            model_id: Unique identifier for the model
            df: Input DataFrame containing both historical data (with 'y') and future data (without 'y')
            periods: Number of periods to forecast (not used if future data is already in df)
            include_weather: Whether to include weather as regressors
            include_events: Whether to include events as regressors
            include_time_features: Whether to include time-based features
            force_retrain: Whether to force model retraining
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Split dataframe into historical (training) and future data for reference
            historical_df = df.dropna(subset=['y']).copy()
            
            # Try to load existing model (unless force_retrain is True)
            model = None if force_retrain else self._load_model(model_id)
            
            # Train new model if not found or force_retrain is True
            if model is None or force_retrain:
                data_logger.info(f"Training new model (model not found or force_retrain={force_retrain})")
                model, forecast = self.train_model(
                    df, 
                    model_id, 
                    periods, 
                    include_weather, 
                    include_events,
                    include_time_features
                )
            else:
                data_logger.info(f"Using existing model: {model_id}")
                
                # Ensure all required features are present in the dataframe
                future_df = df.copy()
                
                # Add time-based features if needed
                if include_time_features:
                    future_df = self.add_time_features(future_df)
                
                # Get list of extra regressors needed by the model
                regressor_names = list(model.extra_regressors.keys())
                
                # Check for missing regressors in the input dataframe
                missing_regressors = [col for col in regressor_names 
                                    if col not in future_df.columns]
                
                if missing_regressors:
                    data_logger.warning(f"Missing regressors in input data: {missing_regressors}")
                    
                    for col in missing_regressors:
                        # Initialize with appropriate default
                        if col.startswith('is_') or col in ['event', 'holiday', 'festival']:
                            # For binary features, use 0
                            future_df[col] = 0
                        elif 'precipitation' in col or 'rain' in col or 'snow' in col:
                            # For precipitation features, use 0
                            future_df[col] = 0
                        else:
                            # For other features, use a reasonable default (0)
                            future_df[col] = 0
                
                # Save the future dataframe to a csv file
                future_df.to_csv(os.path.join(self.model_dir, f"loaded_model_{model_id}_future_dataframe.csv"), index=False)

                # Make forecast
                forecast = model.predict(future_df)
            
            if model is None:
                return {
                    "success": False,
                    "error": "Failed to create or load model"
                }
                
            # Merge with original data
            result = pd.merge(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                df[['ds', 'y']],
                on='ds',
                how='left'
            )
            
            # Calculate accuracy metrics for historical period
            historical = result.dropna(subset=['y'])
            
            if len(historical) > 0:
                mape = np.mean(np.abs((historical['y'] - historical['yhat']) / historical['y'])) * 100
                rmse = np.sqrt(np.mean((historical['y'] - historical['yhat']) ** 2))
            else:
                mape = None
                rmse = None
            
            # Split into historical and forecast periods
            max_historical_date = historical_df['ds'].max()
            historical_data = result[result['ds'] <= max_historical_date]
            forecast_data = result[result['ds'] > max_historical_date]
            
            # Convert to lists for JSON serialization
            dates = result['ds'].dt.strftime('%Y-%m-%d').tolist()
            actuals = result['y'].tolist()
            predictions = result['yhat'].tolist()
            lower_bound = result['yhat_lower'].tolist()
            upper_bound = result['yhat_upper'].tolist()
            
            # Extract model components
            components = {
                "trend": forecast['trend'].tolist(),
                "weekly": forecast['weekly'].tolist() if 'weekly' in forecast.columns else None,
                "yearly": forecast['yearly'].tolist() if 'yearly' in forecast.columns else None
            }
            
            # Add weather components if available
            weather_components = {}
            for col in forecast.columns:
                if col.startswith('extra_regressors_'):
                    # Extract regressor name from column name
                    regressor_name = col.replace('extra_regressors_', '')
                    if regressor_name in df.columns:
                        weather_components[regressor_name] = forecast[col].tolist()
            
            # Add weather components to the response if any were added
            if weather_components:
                components["weather"] = weather_components
            
            # Create forecast response
            response = {
                "success": True,
                "model_id": model_id,
                "dates": dates,
                "actuals": actuals,
                "predictions": predictions,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "metrics": {
                    "mape": mape,
                    "rmse": rmse
                },
                "components": components,
                "historical_points": len(historical_data),
                "forecast_points": len(forecast_data)
            }
            
            data_logger.info(f"Generated forecast for model: {model_id}")
            return response
            
        except Exception as e:
            data_logger.error(f"Error generating forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }



    def _save_model(self, model: Prophet, model_id: str) -> bool:
        """
        Save a Prophet model
        
        Args:
            model: Prophet model
            model_id: Unique identifier for the model
            
        Returns:
            Success flag
        """
        try:
            # Serialize the model to JSON
            model_json = model_to_json(model)
            
            if self.use_s3:
                # Save to S3
                s3_key = f"{settings.S3_MODEL_PREFIX}{model_id}.json"
                
                # Convert model_json to string if it's not already
                if isinstance(model_json, dict):
                    model_json = json.dumps(model_json)
                
                # Use put_object directly with string body
                self.s3_service.s3_client.put_object(
                    Bucket=self.s3_service.bucket_name,
                    Key=s3_key,
                    Body=model_json
                )
                
                data_logger.info(f"Saved model weights to S3: {s3_key}")
            else:
                # Save to local filesystem
                os.makedirs(self.model_dir, exist_ok=True)
                model_path = os.path.join(self.model_dir, f"{model_id}.json")
                
                # Serialize and save the model
                with open(model_path, 'w') as f:
                    json.dump(model_json, f)
                    
                data_logger.info(f"Saved model weights to: {model_path}")
                
            return True
            
        except Exception as e:
            data_logger.error(f"Error saving model {model_id}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def _load_model(self, model_id: str) -> Optional[Prophet]:
        """
        Load a Prophet model
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Prophet model (or None if not found)
        """
        try:
            if self.use_s3:
                # Load from S3
                s3_key = f"{settings.S3_MODEL_PREFIX}{model_id}.json"
                
                if not self.s3_service.file_exists(s3_key):
                    data_logger.warning(f"Model not found in S3: {s3_key}")
                    return None
                
                # Get the model JSON from S3
                response = self.s3_service.s3_client.get_object(
                    Bucket=self.s3_service.bucket_name,
                    Key=s3_key
                )
                model_json = response['Body'].read().decode('utf-8')
                
                # Load the model from JSON
                model = model_from_json(model_json)
                data_logger.info(f"Loaded model from S3: {s3_key}")
                return model
            else:
                # Load from local filesystem
                model_path = os.path.join(self.model_dir, f"{model_id}.json")
                
                if not os.path.exists(model_path):
                    data_logger.warning(f"Model not found: {model_path}")
                    return None
                
                # Load the model
                with open(model_path, 'r') as f:
                    model_json = json.load(f)
                
                model = model_from_json(model_json)
                data_logger.info(f"Loaded model from: {model_path}")
                return model
            
        except Exception as e:
            data_logger.error(f"Error loading model {model_id}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None

def test_prophet_model():
    """Test the ProphetModel functionality"""
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from app.data.loader import DataLoader
    from app.data.processor import DataProcessor
    from app.services.events_service import EventsService
    from app.services.weather_service import WeatherService
    
    # Load and process data
    loader = DataLoader()
    processor = DataProcessor()
    
    company = "forge"
    raw_data = loader.load_company_data(company)
    processed_data = processor.process_company_data(company, raw_data)
    
    # Create test dataframe
    if "daily_sales" in processed_data:
        df = processed_data["daily_sales"]
        
        # Initialize Prophet model
        model = ProphetModel()
        
        # Create Prophet dataframe
        prophet_df = model.prepare_data_for_prophet(df, "total_revenue")
        
        # Get events data
        events_service = EventsService()
        
        # Get date range
        min_date = df["date"].min().strftime("%Y-%m-%d")
        max_date = df["date"].max().strftime("%Y-%m-%d")
        
        # Prepare events data
        events_df = events_service.prepare_events_for_prophet(min_date, max_date)
        
        # Add events features
        prophet_df = model.add_events_features(prophet_df, events_df)

        # Prepare weather data
        weather_service = WeatherService()
        weather_df = weather_service.prepare_weather_for_prophet(min_date, max_date)
        
        # Add weather features
        prophet_df = model.add_weather_features(prophet_df, weather_df)
        
        # Add time features
        prophet_df = model.add_time_features(prophet_df)

        #print(f"Prophet dataframe: {prophet_df.head()}")
        
        # Create directory if it doesn't exist
        data_dir = os.path.join("data", company, "revenue")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save prophet dataframe
        prophet_df.to_csv(os.path.join(data_dir, f"{company}_prophet_revenue_data.csv"), index=False)
        print(f"Prophet dataframe saved to: {os.path.join(data_dir, f'{company}_prophet_revenue_data.csv')}")

        # Test model training with time features
        model_id = f"{company}_revenue"
        print(f"Training model: {model_id}")
        _, forecast = model.train_model(
            prophet_df, 
            model_id, 
            periods=15,
            include_weather=True,
            include_events=True,
            include_time_features=True
        )
        
        # Test model loading
        print(f"Testing model loading: {model_id}")
        loaded_model = model._load_model(model_id)
        print(f"Model loaded successfully: {loaded_model is not None}")
        
        # Test forecast generation with loaded model
        print(f"Generating forecast with loaded model")
        forecast_result = model.generate_forecast(
            model_id, 
            prophet_df, 
            periods=5,
            include_weather=True,
            include_events=True,
            include_time_features=True,
            force_retrain=False  # Use saved model
        )
        
        print(f"Forecast success: {forecast_result['success']}")
        if forecast_result['metrics']['mape'] is not None:
            print(f"MAPE: {forecast_result['metrics']['mape']:.2f}%")
        print(f"Forecast periods: {len(forecast_result['dates']) - len(prophet_df)}")
        
        # Test force retrain
        print(f"Testing force retrain")
        forecast_result_retrained = model.generate_forecast(
            model_id, 
            prophet_df, 
            periods=5,
            include_weather=True,
            include_events=True,
            include_time_features=True,
            force_retrain=True  # Force retrain
        )
        
        print(f"Retrained forecast success: {forecast_result_retrained['success']}")
        
        return "Prophet model test completed successfully"
    else:
        return "No daily sales data available for testing"

if __name__ == "__main__":
    test_prophet_model()