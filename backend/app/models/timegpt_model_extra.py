import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import traceback
import json
import requests
from nixtla import TimeGPT
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.logger import data_logger
from app.core.config import settings

class TimeGPTModel:
    """
    TimeGPT model for multivariate time series forecasting with external regressors
    """
    
    def __init__(self, model_dir: str = "data/models", cache_dir: str = "data/cache"):
        """Initialize the TimeGPT model"""
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        # Initialize TimeGPT with your API key
        self.api_key = settings.TIMEGPT_API_KEY
        self.timegpt = TimeGPT(api_key=self.api_key)
        data_logger.info(f"TimeGPTModel initialized with model directory: {model_dir}")
    
    def prepare_data_for_timegpt(self, df: pd.DataFrame, target_col: str, 
                                date_col: str = "date") -> pd.DataFrame:
        """
        Prepare data for TimeGPT model
        
        Args:
            df: Input DataFrame with date column
            target_col: Target column for forecasting
            date_col: Date column name
            
        Returns:
            DataFrame formatted for TimeGPT
        """
        try:
            # Create TimeGPT format dataframe
            timegpt_df = df[[date_col, target_col]].copy()
            
            # Rename columns to TimeGPT's required format - 'ds' for date and 'y' for target
            timegpt_df.columns = ["ds", "y"]
            
            # Sort by date
            timegpt_df = timegpt_df.sort_values("ds")
            
            # Ensure date column is datetime
            timegpt_df["ds"] = pd.to_datetime(timegpt_df["ds"])
            
            data_logger.info(f"Prepared data for TimeGPT with {len(timegpt_df)} rows")
            return timegpt_df
            
        except Exception as e:
            data_logger.error(f"Error preparing data for TimeGPT: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the data
        
        Args:
            df: DataFrame with 'ds' datetime column
            
        Returns:
            DataFrame with time features added
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Ensure ds is datetime
            result_df['ds'] = pd.to_datetime(result_df['ds'])
            
            # Add time-based features if they don't exist
            if 'dayofweek' not in result_df.columns:
                result_df['dayofweek'] = result_df['ds'].dt.dayofweek
            
            if 'is_weekend' not in result_df.columns:
                result_df['is_weekend'] = (result_df['ds'].dt.dayofweek >= 5).astype(int)
            
            if 'month' not in result_df.columns:
                result_df['month'] = result_df['ds'].dt.month
            
            if 'year' not in result_df.columns:
                result_df['year'] = result_df['ds'].dt.year
            
            data_logger.info(f"Added time features to data with {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            data_logger.error(f"Error adding time features: {str(e)}")
            data_logger.error(traceback.format_exc())
            return df
    
    def add_events_features(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add events features to the data
        
        Args:
            df: DataFrame with 'ds' datetime column
            events_df: DataFrame with events data
            
        Returns:
            DataFrame with events features added
        """
        try:
            # Return original if events_df is empty
            if events_df.empty:
                return df
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Merge events with main dataframe
            result_df = pd.merge(result_df, events_df, on='ds', how='left')
            
            # Fill NaN values
            for col in ['event', 'holiday', 'festival']:
                if col in result_df.columns:
                    result_df[col] = result_df[col].fillna(0).astype(int)
            
            data_logger.info(f"Added events features to data with {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            data_logger.error(f"Error adding events features: {str(e)}")
            data_logger.error(traceback.format_exc())
            return df
    
    def add_weather_features(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weather features to the data
        
        Args:
            df: DataFrame with 'ds' datetime column
            weather_df: DataFrame with weather data
            
        Returns:
            DataFrame with weather features added
        """
        try:
            # Return original if weather_df is empty
            if weather_df.empty:
                return df
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Merge weather with main dataframe
            result_df = pd.merge(result_df, weather_df, on='ds', how='left')
            
            # Forward fill weather data (use the most recent available data)
            weather_cols = ['temperature', 'precipitation', 'rainy', 'sunny', 'temperature_squared']
            for col in weather_cols:
                if col in result_df.columns:
                    result_df[col] = result_df[col].fillna(method='ffill')
            
            data_logger.info(f"Added weather features to data with {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            data_logger.error(f"Error adding weather features: {str(e)}")
            data_logger.error(traceback.format_exc())
            return df
    
    def train_model(self, 
                    df: pd.DataFrame, 
                    model_id: str,
                    periods: int = 30,
                    include_weather: bool = True,
                    include_events: bool = True,
                    include_time_features: bool = True,
                    fine_tune: bool = False,
                    loss_function: str = 'RMSE') -> Tuple[Any, pd.DataFrame]:
        """
        Train a TimeGPT model and generate forecast
        
        Args:
            df: DataFrame with historical data in TimeGPT format
            model_id: Unique identifier for the model
            periods: Number of periods to forecast
            include_weather: Whether to include weather features
            include_events: Whether to include events features
            include_time_features: Whether to include time features
            fine_tune: Whether to fine-tune the model
            loss_function: Loss function for fine-tuning
            
        Returns:
            Tuple of model and forecast DataFrame
        """
        try:
            data_logger.info(f"Training TimeGPT model: {model_id}")
            
            # Prepare exogenous variables
            exogenous_variables = []
            
            if include_time_features:
                exogenous_variables.extend(['dayofweek', 'is_weekend', 'month', 'year'])
            
            if include_weather:
                exogenous_variables.extend(['temperature', 'precipitation', 'rainy', 'sunny', 'temperature_squared'])
            
            if include_events:
                exogenous_variables.extend(['event', 'holiday', 'festival'])
            
            # Keep only exogenous variables that exist in the dataframe
            exogenous_variables = [var for var in exogenous_variables if var in df.columns]
            
            # Split data for training and testing (last 30 days as test)
            train_df = df.copy()
            
            # Fine-tune if requested
            if fine_tune and len(train_df) >= 90:  # Need enough data for fine-tuning
                # Calculate training and validation split
                train_size = int(len(train_df) * 0.8)
                finetune_train = train_df.iloc[:train_size]
                finetune_val = train_df.iloc[train_size:]
                
                data_logger.info(f"Fine-tuning TimeGPT model on {len(finetune_train)} samples with {loss_function} loss")
                
                # Fine-tune the model
                finetune_model = self.timegpt.finetune(
                    df=finetune_train, 
                    val_df=finetune_val,
                    id=model_id,
                    horizon=periods,
                    loss=loss_function,
                    exogenous_variables=exogenous_variables if exogenous_variables else None
                )
                
                # Save the fine-tuned model ID for later use
                self.save_model(model_id, {"finetune_id": model_id})
                model = finetune_model
            else:
                # Use regular model with exogenous variables
                model = None  # TimeGPT doesn't need a model object for forecasting
            
            # Generate future dataframe for prediction
            last_date = df['ds'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            # Create future dataframe with exogenous variables
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Add exogenous variables to future_df if they exist
            if include_time_features:
                future_df['dayofweek'] = future_df['ds'].dt.dayofweek
                future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
                future_df['month'] = future_df['ds'].dt.month
                future_df['year'] = future_df['ds'].dt.year
            
            # Combine historical and future data for prediction
            forecast_df = pd.concat([df[['ds', 'y']], future_df[['ds']]])
            forecast_df = forecast_df.reset_index(drop=True)
            
            # If using fine-tuned model
            if fine_tune and len(train_df) >= 90:
                forecast_result = self.timegpt.forecast(
                    df=df,
                    h=periods,
                    finetune_id=model_id,
                    exogenous_variables=exogenous_variables if exogenous_variables else None,
                    exogenous_df=future_df if exogenous_variables else None
                )
            else:
                # Generate forecast using TimeGPT
                forecast_result = self.timegpt.forecast(
                    df=df,
                    h=periods,
                    exogenous_variables=exogenous_variables if exogenous_variables else None,
                    exogenous_df=future_df if exogenous_variables else None
                )
            
            # Merge forecast with original data
            forecast = pd.concat([df[['ds', 'y']], forecast_result[['ds', 'TimeGPT']]])
            forecast = forecast.rename(columns={'TimeGPT': 'yhat'})
            
            # Save model information
            model_info = {
                "model_id": model_id,
                "exogenous_variables": exogenous_variables,
                "fine_tuned": fine_tune,
                "trained_at": datetime.now().isoformat()
            }
            self.save_model(model_id, model_info)
            
            data_logger.info(f"Successfully trained TimeGPT model: {model_id}")
            
            return model, forecast
            
        except Exception as e:
            data_logger.error(f"Error training TimeGPT model: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None, pd.DataFrame()
    
    def generate_forecast(self,
                         model_id: str,
                         df: pd.DataFrame,
                         periods: int = 30,
                         include_weather: bool = True,
                         include_events: bool = True,
                         include_time_features: bool = True,
                         force_retrain: bool = False,
                         fine_tune: bool = False,
                         loss_function: str = 'RMSE') -> Dict[str, Any]:
        """
        Generate a forecast using TimeGPT
        
        Args:
            model_id: Unique identifier for the model
            df: DataFrame with historical data
            periods: Number of periods to forecast
            include_weather: Whether to include weather features
            include_events: Whether to include events features
            include_time_features: Whether to include time features
            force_retrain: Whether to force retraining of the model
            fine_tune: Whether to fine-tune the model
            loss_function: Loss function for fine-tuning
            
        Returns:
            Dictionary with forecast results
        """
        try:
            data_logger.info(f"Generating forecast with TimeGPT model: {model_id}")
            
            # Check if we need to train a new model
            model_exists = os.path.exists(os.path.join(self.model_dir, f"{model_id}.json"))
            if not model_exists or force_retrain:
                # Train a new model
                model, forecast = self.train_model(
                    df,
                    model_id,
                    periods,
                    include_weather,
                    include_events,
                    include_time_features,
                    fine_tune,
                    loss_function
                )
            else:
                # Load existing model
                model_info = self.load_model(model_id)
                
                if not model_info:
                    data_logger.warning(f"Failed to load model info for {model_id}, retraining")
                    model, forecast = self.train_model(
                        df,
                        model_id,
                        periods,
                        include_weather,
                        include_events,
                        include_time_features,
                        fine_tune,
                        loss_function
                    )
                else:
                    data_logger.info(f"Using existing TimeGPT model: {model_id}")
                    
                    # Prepare exogenous variables from model info
                    exogenous_variables = model_info.get("exogenous_variables", [])
                    
                    # Generate future dataframe for prediction
                    last_date = df['ds'].max()
                    future_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=periods,
                        freq='D'
                    )
                    
                    # Create future dataframe with exogenous variables
                    future_df = pd.DataFrame({'ds': future_dates})
                    
                    # Add exogenous variables to future_df if they exist
                    if include_time_features and any(x in exogenous_variables for x in ['dayofweek', 'is_weekend', 'month', 'year']):
                        future_df['dayofweek'] = future_df['ds'].dt.dayofweek
                        future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
                        future_df['month'] = future_df['ds'].dt.month
                        future_df['year'] = future_df['ds'].dt.year
                    
                    # If using fine-tuned model
                    if model_info.get("fine_tuned", False) and "finetune_id" in model_info:
                        forecast_result = self.timegpt.forecast(
                            df=df,
                            h=periods,
                            finetune_id=model_info["finetune_id"],
                            exogenous_variables=exogenous_variables if exogenous_variables else None,
                            exogenous_df=future_df if exogenous_variables else None
                        )
                    else:
                        # Generate forecast using TimeGPT
                        forecast_result = self.timegpt.forecast(
                            df=df,
                            h=periods,
                            exogenous_variables=exogenous_variables if exogenous_variables else None,
                            exogenous_df=future_df if exogenous_variables else None
                        )
                    
                    # Merge forecast with original data
                    forecast = pd.concat([df[['ds', 'y']], forecast_result[['ds', 'TimeGPT']]])
                    forecast = forecast.rename(columns={'TimeGPT': 'yhat'})
            
            # Generate prediction intervals
            forecast['yhat_lower'] = forecast['yhat'] * 0.9  # Approximate 90% lower bound
            forecast['yhat_upper'] = forecast['yhat'] * 1.1  # Approximate 90% upper bound
            
            # Calculate metrics on historical data
            historical_data = forecast[forecast['y'].notna()]
            metrics = {}
            
            if len(historical_data) > 0:
                # Calculate MAPE
                actuals = historical_data['y'].values
                predictions = historical_data['yhat'].values
                abs_diff = np.abs(actuals - predictions)
                mape = np.mean(abs_diff / np.abs(actuals)) * 100 if np.sum(np.abs(actuals)) > 0 else None
                metrics['mape'] = mape
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
                metrics['rmse'] = rmse
                
                # Calculate MAE
                mae = np.mean(abs_diff)
                metrics['mae'] = mae
            else:
                metrics = {
                    'mape': None,
                    'rmse': None,
                    'mae': None
                }
            
            # Prepare result
            result = {
                "success": True,
                "model_id": model_id,
                "dates": forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                "actuals": forecast['y'].fillna(np.nan).tolist(),
                "predictions": forecast['yhat'].tolist(),
                "lower_bounds": forecast['yhat_lower'].tolist(),
                "upper_bounds": forecast['yhat_upper'].tolist(),
                "metrics": metrics
            }
            
            # Cache forecast results
            self._cache_forecast(model_id, result)
            
            data_logger.info(f"Successfully generated forecast with TimeGPT model: {model_id}")
            
            return result
            
        except Exception as e:
            data_logger.error(f"Error generating forecast with TimeGPT: {str(e)}")
            data_logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "model_id": model_id
            }
    
    def cross_validate(self, 
                      df: pd.DataFrame, 
                      model_id: str,
                      cv_params: Dict[str, Any] = None,
                      exogenous_variables: List[str] = None) -> Dict[str, Any]:
        """
        Perform cross-validation for a TimeGPT model
        
        Args:
            df: DataFrame with historical data
            model_id: Unique identifier for the model
            cv_params: Parameters for cross-validation
            exogenous_variables: List of exogenous variables
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            data_logger.info(f"Performing cross-validation for TimeGPT model: {model_id}")
            
            if cv_params is None:
                cv_params = {
                    "n_windows": 3,
                    "window_size": 30
                }
            
            # Ensure df has at least 90 days for meaningful cross-validation
            if len(df) < 90:
                data_logger.warning(f"Not enough data for cross-validation (needed: 90, got: {len(df)})")
                return {
                    "success": False,
                    "error": "Not enough data for cross-validation"
                }
            
            # Perform cross-validation using TimeGPT's built-in function
            cv_results = self.timegpt.cross_validation(
                df=df,
                h=cv_params.get("window_size", 30),
                n_windows=cv_params.get("n_windows", 3),
                exogenous_variables=exogenous_variables
            )
            
            # Calculate aggregate metrics
            metrics = {
                "mape": cv_results['MAPE'].mean(),
                "rmse": cv_results['RMSE'].mean(),
                "mae": cv_results['MAE'].mean()
            }
            
            result = {
                "success": True,
                "model_id": model_id,
                "metrics": metrics,
                "cv_results": cv_results.to_dict()
            }
            
            data_logger.info(f"Cross-validation for TimeGPT model {model_id} completed with MAPE: {metrics['mape']:.2f}%")
            
            return result
            
        except Exception as e:
            data_logger.error(f"Error performing cross-validation for TimeGPT: {str(e)}")
            data_logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "model_id": model_id
            }
    
    def save_model(self, model_id: str, model_info: Dict[str, Any]) -> bool:
        """
        Save model information to file
        
        Args:
            model_id: Unique identifier for the model
            model_info: Dictionary with model information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.model_dir, f"{model_id}.json")
            
            with open(file_path, 'w') as f:
                json.dump(model_info, f)
            
            data_logger.info(f"Saved TimeGPT model info to: {file_path}")
            return True
            
        except Exception as e:
            data_logger.error(f"Error saving TimeGPT model info: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def load_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load model information from file
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Dictionary with model information
        """
        try:
            file_path = os.path.join(self.model_dir, f"{model_id}.json")
            
            if not os.path.exists(file_path):
                data_logger.warning(f"Model file not found: {file_path}")
                return None
            
            with open(file_path, 'r') as f:
                model_info = json.load(f)
            
            data_logger.info(f"Loaded TimeGPT model info from: {file_path}")
            return model_info
            
        except Exception as e:
            data_logger.error(f"Error loading TimeGPT model info: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None
    
    def _cache_forecast(self, model_id: str, forecast: Dict[str, Any]) -> bool:
        """
        Cache forecast results to file
        
        Args:
            model_id: Unique identifier for the model
            forecast: Dictionary with forecast results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.cache_dir, f"forecast_timegpt_{model_id}.json")
            
            with open(file_path, 'w') as f:
                json.dump(forecast, f)
            
            data_logger.info(f"Cached TimeGPT forecast to: {file_path}")
            return True
            
        except Exception as e:
            data_logger.error(f"Error caching TimeGPT forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def get_cached_forecast(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached forecast results from file
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Dictionary with forecast results
        """
        try:
            file_path = os.path.join(self.cache_dir, f"forecast_timegpt_{model_id}.json")
            
            if not os.path.exists(file_path):
                data_logger.warning(f"Cached forecast not found: {file_path}")
                return None
            
            with open(file_path, 'r') as f:
                forecast = json.load(f)
            
            data_logger.info(f"Loaded cached TimeGPT forecast from: {file_path}")
            return forecast
            
        except Exception as e:
            data_logger.error(f"Error loading cached TimeGPT forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None 