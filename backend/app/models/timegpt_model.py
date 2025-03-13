# app/models/timegpt_model.py
import os
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import traceback
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.logger import data_logger

class TimeGPTModel:
    """
    TimeGPT model for multivariate time series forecasting with external regressors
    
    This implementation is structured similarly to ProphetModel for easier integration
    """
    
    def __init__(self, model_dir: str = "data/models", api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, model: str = "timegpt-1"):
        """Initialize the TimeGPT model"""
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize TimeGPT client
        self.api_key = api_key
        self.model_type = model
        
        try:
            # Initialize Nixtla client
            client_args = {"api_key": api_key}
            if base_url:
                client_args["base_url"] = base_url
                
            self.client = NixtlaClient(**client_args)
            data_logger.info(f"TimeGPTModel initialized with model type: {model} and model directory: {model_dir}")
        except Exception as e:
            data_logger.error(f"Error initializing TimeGPT client: {str(e)}")
            data_logger.error(traceback.format_exc())
            raise
    
    def prepare_data_for_timegpt(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Prepare data for TimeGPT model
        
        Args:
            df: Input DataFrame with 'date' column
            target_col: Target column for forecasting
            
        Returns:
            DataFrame formatted for TimeGPT
        """
        try:
            # Create TimeGPT format dataframe, matching Prophet's format
            timegpt_df = df[["date", target_col]].copy()
            
            # Rename columns to TimeGPT's required format (which matches Prophet)
            timegpt_df.columns = ["ds", "y"]
            
            # Sort by date
            timegpt_df = timegpt_df.sort_values("ds")
            
            # Ensure date column is datetime
            timegpt_df["ds"] = pd.to_datetime(timegpt_df["ds"])
            
            # Add time-based features if they exist in the original dataframe (matching Prophet)
            for feature in ['dayofweek', 'is_weekend', 'month', 'year']:
                if feature in df.columns:
                    timegpt_df[feature] = df[feature].values
            
            data_logger.info(f"Prepared data for TimeGPT with {len(timegpt_df)} rows")
            return timegpt_df
            
        except Exception as e:
            data_logger.error(f"Error preparing data for TimeGPT: {str(e)}")
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
                events_df[["ds", "event", "holiday", "festival"]],
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
    
    def add_weather_features(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weather features to the input DataFrame
        
        Args:
            df: Input DataFrame
            weather_df: Weather DataFrame
            
        Returns:
            DataFrame with added weather features
        """
        try:
            if weather_df.empty:
                data_logger.warning("No weather data provided")
                return df
                
            # Copy input DataFrame
            result_df = df.copy()
            
            # Merge with weather data
            result_df = pd.merge(
                result_df,
                weather_df[["ds", "temperature", "precipitation", "rainy", "sunny"]],
                on="ds",
                how="left"
            )
            
            # Fill NaN values with reasonable defaults
            result_df["temperature"] = result_df["temperature"].fillna(result_df["temperature"].mean())
            result_df["precipitation"] = result_df["precipitation"].fillna(0)
            result_df["rainy"] = result_df["rainy"].fillna(0)
            result_df["sunny"] = result_df["sunny"].fillna(0)
            
            data_logger.info(f"Added weather features")
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
            
            # Add time-based features if not already present
            if 'dayofweek' not in result_df.columns:
                result_df['dayofweek'] = result_df['ds'].dt.dayofweek
            if 'month' not in result_df.columns:
                result_df['month'] = result_df['ds'].dt.month
            if 'year' not in result_df.columns:
                result_df['year'] = result_df['ds'].dt.year
            if 'is_weekend' not in result_df.columns:
                result_df['is_weekend'] = (result_df['ds'].dt.dayofweek >= 5).astype(int)
            
            data_logger.info(f"Added time-based features")
            return result_df
            
        except Exception as e:
            data_logger.error(f"Error adding time features: {str(e)}")
            data_logger.error(traceback.format_exc())
            return df
    
    def _get_exogenous_variables(self, df: pd.DataFrame, 
                               include_weather: bool = True,
                               include_events: bool = True,
                               include_time_features: bool = True) -> List[str]:
        """
        Get list of exogenous variables from DataFrame based on inclusion flags
        
        Args:
            df: Input DataFrame
            include_weather: Whether to include weather features
            include_events: Whether to include events features
            include_time_features: Whether to include time features
            
        Returns:
            List of exogenous variable names
        """
        exogenous_variables = []
        
        if include_time_features:
            time_vars = ['dayofweek', 'is_weekend', 'month', 'year']
            exogenous_variables.extend([var for var in time_vars if var in df.columns])
            
        if include_weather:
            weather_vars = ['temperature', 'precipitation', 'rainy', 'sunny']
            exogenous_variables.extend([var for var in weather_vars if var in df.columns])
            
        if include_events:
            event_vars = ['event', 'holiday', 'festival']
            exogenous_variables.extend([var for var in event_vars if var in df.columns])
            
        return exogenous_variables
    
    def train_model(self, 
                   df: pd.DataFrame,
                   model_id: str,
                   freq: str = 'D',
                   periods: int = 30,
                   include_weather: bool = True,
                   include_events: bool = True,
                   include_time_features: bool = True,
                   finetune: bool = True,
                   finetune_steps: int = 10,
                   finetune_loss: str = "mse",
                   finetune_depth: int = 2) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Train a TimeGPT model with regressors
        
        Args:
            df: Input DataFrame in TimeGPT format (with ds, y)
            model_id: Unique identifier for the model
            periods: Number of periods to forecast
            include_weather: Whether to include weather as regressors
            include_events: Whether to include events as regressors
            include_time_features: Whether to include time-based features as regressors
            finetune: Whether to fine-tune the model
            finetune_steps: Number of fine-tuning steps
            finetune_loss: Loss function for fine-tuning ('mse', 'mae', 'mape', 'smape')
            finetune_depth: Level of fine-tuning (1-5)
            
        Returns:
            Tuple of (model metadata dict, forecast DataFrame)
        """
        try:
            data_logger.info(f"Training TimeGPT model: {model_id}")
            
            # Get exogenous variables
            exogenous_variables = self._get_exogenous_variables(
                df, include_weather, include_events, include_time_features
            )

            # 1. Convert all dates to standard ISO format strings and back to datetime
            df['ds'] = pd.to_datetime(pd.DatetimeIndex(df['ds']).strftime('%Y-%m-%d'))
            
            # 2. Ensure we have the correct frequency type
            if freq != 'D':
                data_logger.warning(f"Forcing frequency to 'D' to avoid memory issues")
                freq = 'D'
            
            # 3. Check date range for sanity
            min_date = df['ds'].min()
            max_date = df['ds'].max()
            date_range_days = (max_date - min_date).days
            data_logger.info(f"Date range: {min_date} to {max_date} ({date_range_days} days)")
            
            # Check for and remove duplicate timestamps
            if df['ds'].duplicated().any():
                duplicates = df['ds'].duplicated().sum()
                df = df.drop_duplicates('ds')
                data_logger.warning(f"Dropped {duplicates} duplicate timestamps")

            # Sort by date
            df = df.sort_values('ds')

            # Check for missing dates in the sequence
            date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
            missing_dates = set(date_range) - set(df['ds'])

            if missing_dates:
                data_logger.warning(f"Found {len(missing_dates)} missing dates in the time series")
                
                # Create DataFrame with missing dates
                missing_df = pd.DataFrame({'ds': list(missing_dates)})
                
                # Fill with median values for numeric columns
                for col in df.columns:
                    if col != 'ds':
                        if pd.api.types.is_numeric_dtype(df[col]):
                            missing_df[col] = df[col].median()
                        else:
                            # For categorical columns, use mode
                            missing_df[col] = df[col].mode()[0] if not df[col].mode().empty else 0
                
                # Combine with original data and sort
                df = pd.concat([df, missing_df]).sort_values('ds')
                data_logger.info(f"Filled {len(missing_dates)} missing dates with median/mode values")

            # Ensure the time column is a proper datetime
            df['ds'] = pd.to_datetime(df['ds'])

            # Log the final dataset stats
            data_logger.info(f"Final dataset has {len(df)} rows with continuous daily frequency")

            '''
            # When calling finetune, ensure finetune_loss is a string
            if isinstance(finetune_loss, int):
                finetune_loss = "mse"  # Default to mse if a number was passed
                data_logger.warning(f"finetune_loss was a number, converted to 'mse'")
                
            data_logger.info(f"Fine-tuning TimeGPT model with '{finetune_loss}' loss and depth {finetune_depth}")
        
            # Ensure finetune_steps is an integer
            if not isinstance(finetune_steps, int):
                try:
                    finetune_steps = int(finetune_steps)
                    data_logger.warning(f"finetune_steps was not an integer, converted to {finetune_steps}")
                except (ValueError, TypeError):
                    finetune_steps = 10  # Default to 10 steps if conversion fails
                    data_logger.warning(f"finetune_steps could not be converted to an integer, using default: {finetune_steps}")

            # Also ensure finetune_depth is an integer
            if not isinstance(finetune_depth, int):
                try:
                    finetune_depth = int(finetune_depth)
                    data_logger.warning(f"finetune_depth was not an integer, converted to {finetune_depth}")
                except (ValueError, TypeError):
                    finetune_depth = 2  # Default to depth 2 if conversion fails
                    data_logger.warning(f"finetune_depth could not be converted to an integer, using default: {finetune_depth}")
            '''
                    
            # Fine-tune the model if requested
            if finetune:
                data_logger.info(f"Fine-tuning TimeGPT model with {finetune_loss} loss and depth {finetune_depth}")
                
                # Fine-tune the model with the specified parameters
                finetuned_model_id = self.client.finetune(
                    df=df,
                    freq=freq,
                    output_model_id=model_id,
                    finetune_steps=finetune_steps,
                    finetune_loss=finetune_loss,
                    finetune_depth=finetune_depth,
                    time_col='ds',
                    target_col='y',
                    model=self.model_type
                )
                
                data_logger.info(f"Successfully fine-tuned TimeGPT model: {finetuned_model_id}")
            else:
                finetuned_model_id = None
                data_logger.info("Skipping fine-tuning")
            
            '''
            if not isinstance(periods, int):
                try:
                    periods = int(periods)
                    if periods < 1:
                        periods = 5
                        data_logger.warning(f"periods was less than 1, converted to 5")
                except (ValueError, TypeError):
                    periods = 5
                    data_logger.warning(f"periods could not be converted to an integer, using default: {periods}")
            '''

            # Create future dataframe for forecast
            future_dates = pd.date_range(
                start=df['ds'].max() + timedelta(days=1),
                periods=periods, 
                freq=freq
            )
            

            # Create future exogenous dataframe with time features
            future_exog = pd.DataFrame({'ds': future_dates})
            
            # Add time features to future exogenous dataframe if needed
            if include_time_features:
                future_exog['dayofweek'] = future_exog['ds'].dt.dayofweek
                future_exog['is_weekend'] = (future_exog['ds'].dt.dayofweek >= 5).astype(int)
                future_exog['month'] = future_exog['ds'].dt.month
                future_exog['year'] = future_exog['ds'].dt.year
            
            # For weather and event features, we'll need to use appropriate forecasting or assumptions
            # Here we use simple defaults for demonstration
            if include_weather:
                # For weather, use mean values from training data
                for weather_var in ['temperature', 'precipitation', 'rainy', 'sunny']:
                    if weather_var in df.columns:
                        future_exog[weather_var] = df[weather_var].mean()
            
            if include_events:
                # For events, initialize with zeros (no events)
                for event_var in ['event', 'holiday', 'festival']:
                    if event_var in df.columns:
                        future_exog[event_var] = 0
            
            # Generate forecast
            forecast_params = {
                'df': df,
                'freq': freq,
                'h': periods,
                'level': [80, 90],  # Confidence intervals
                'time_col': 'ds',
                'target_col': 'y',
                'model': self.model_type
            }
            
            # Add future exogenous data if we have any
            if exogenous_variables:
                forecast_params['X_df'] = future_exog[['ds'] + exogenous_variables]
            
            # Add fine-tuned model ID if available
            if finetuned_model_id:
                forecast_params['finetuned_model_id'] = finetuned_model_id
            
            # Generate forecast
            forecast_result = self.client.forecast(**forecast_params)
            
            # Convert forecast to format similar to Prophet
            forecast = forecast_result.copy()
            forecast = forecast.rename(columns={
                'TimeGPT': 'yhat',
                'TimeGPT-lo-80': 'yhat_lower',
                'TimeGPT-hi-80': 'yhat_upper'
            })
            
            # Create model metadata
            model_metadata = {
                'model_id': model_id,
                'finetuned_model_id': finetuned_model_id,
                'exogenous_variables': exogenous_variables,
                'finetune': finetune,
                'finetune_steps': finetune_steps,
                'finetune_loss': finetune_loss,
                'finetune_depth': finetune_depth,
                'model_type': self.model_type,
                'created_at': datetime.now().isoformat()
            }
            
            # Save model metadata
            self._save_model_metadata(model_metadata, model_id)
            
            data_logger.info(f"TimeGPT model trained successfully: {model_id}")
            return model_metadata, forecast
            
        except Exception as e:
            data_logger.error(f"Error training TimeGPT model: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None, pd.DataFrame()
    
    def load_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Load model metadata
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Model metadata dictionary
        """
        try:
            metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
            
            if not os.path.exists(metadata_path):
                data_logger.warning(f"Model metadata not found: {metadata_path}")
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            data_logger.info(f"Successfully loaded model metadata: {model_id}")
            return metadata
            
        except Exception as e:
            data_logger.error(f"Error loading model metadata {model_id}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None
    
    def _save_model_metadata(self, metadata: Dict[str, Any], model_id: str) -> bool:
        """
        Save model metadata
        
        Args:
            metadata: Model metadata dictionary
            model_id: Unique identifier for the model
            
        Returns:
            Success flag
        """
        try:
            # Create model directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            metadata_path = os.path.join(self.model_dir, f"{model_id}_metadata.json")
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            data_logger.info(f"Saved model metadata to: {metadata_path}")
            return True
            
        except Exception as e:
            data_logger.error(f"Error saving model metadata {model_id}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    '''
    def _prepare_future_exog_df(self, df: pd.DataFrame, periods: int, 
                              exogenous_variables: List[str]) -> pd.DataFrame:
        """
        Prepare future exogenous dataframe for forecasting
        
        Args:
            df: Input DataFrame
            periods: Number of periods to forecast
            exogenous_variables: List of exogenous variable names
            
        Returns:
            DataFrame with future exogenous variables
        """
        try:


            # Create future date range
            future_dates = pd.date_range(
                start=df['ds'].max() + timedelta(days=1),
                periods=periods, 
                freq='D'
            )
            
            # Create future exogenous dataframe
            future_exog = pd.DataFrame({'ds': future_dates})
            
            # Add time features
            time_vars = ['dayofweek', 'is_weekend', 'month', 'year']
            time_vars_to_add = [var for var in time_vars if var in exogenous_variables]
            
            if time_vars_to_add:
                future_exog['dayofweek'] = future_exog['ds'].dt.dayofweek
                future_exog['is_weekend'] = (future_exog['ds'].dt.dayofweek >= 5).astype(int)
                future_exog['month'] = future_exog['ds'].dt.month
                future_exog['year'] = future_exog['ds'].dt.year
            
            # For other exogenous variables, use mean values or defaults
            for var in exogenous_variables:
                if var not in time_vars_to_add and var not in future_exog.columns:
                    if var in df.columns:
                        # Use mean for numeric features
                        if np.issubdtype(df[var].dtype, np.number):
                            future_exog[var] = df[var].mean()
                        else:
                            # Use most common value for categorical
                            future_exog[var] = df[var].mode()[0]
            
            return future_exog
            
        except Exception as e:
            data_logger.error(f"Error preparing future exogenous dataframe: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame({'ds': future_dates})
    '''



    def _prepare_future_exog_df(self, df: pd.DataFrame, periods: int, 
                            exogenous_variables: List[str]) -> pd.DataFrame:
        """
        Prepare future exogenous dataframe for forecasting with enhanced handling
        
        Args:
            df: Input DataFrame
            periods: Number of periods to forecast
            exogenous_variables: List of exogenous variable names
            
        Returns:
            DataFrame with future exogenous variables
        """
        try:
            # Create future date range
            future_dates = pd.date_range(
                start=df['ds'].max() + timedelta(days=1),
                periods=periods, 
                freq='D'
            )
            
            # Create future exogenous dataframe
            future_exog = pd.DataFrame({'ds': future_dates})
            
            # Add time features
            time_vars = ['dayofweek', 'is_weekend', 'month', 'year']
            time_vars_to_add = [var for var in time_vars if var in exogenous_variables]
            
            if time_vars_to_add:
                future_exog['dayofweek'] = future_exog['ds'].dt.dayofweek
                future_exog['is_weekend'] = (future_exog['ds'].dt.dayofweek >= 5).astype(int)
                future_exog['month'] = future_exog['ds'].dt.month
                future_exog['year'] = future_exog['ds'].dt.year
            
            # For weather variables, use a more sophisticated approach
            weather_vars = ['temperature', 'precipitation', 'rainy', 'sunny']
            weather_vars_to_add = [var for var in weather_vars if var in exogenous_variables]
            
            if weather_vars_to_add:
                data_logger.info(f"Adding weather variables for future periods: {weather_vars_to_add}")
                
                # Get the last 30 days of data (or whatever is available)
                recent_days = min(30, len(df))
                recent_data = df.sort_values('ds').tail(recent_days)
                
                for var in weather_vars_to_add:
                    if var in df.columns:
                        # For temperature, consider seasonality by using same days from previous years
                        if var == 'temperature' and len(df) > 365:
                            # Find data from same dates last year if available
                            seasonal_temps = []
                            for future_date in future_dates:
                                same_day_last_year = future_date - timedelta(days=365)
                                same_day_values = df[df['ds'].dt.month == same_day_last_year.month]
                                same_day_values = same_day_values[same_day_values['ds'].dt.day == same_day_last_year.day]
                                
                                if not same_day_values.empty:
                                    seasonal_temps.append(same_day_values[var].values[0])
                                else:
                                    # If not available, use mean from recent data
                                    seasonal_temps.append(recent_data[var].mean())
                                    
                            future_exog[var] = seasonal_temps
                        else:
                            # Use mean values from recent data for other weather variables
                            future_exog[var] = recent_data[var].mean()
            
            # For events variables, use a more intelligent approach
            event_vars = ['event', 'holiday', 'festival']
            event_vars_to_add = [var for var in event_vars if var in exogenous_variables]
            
            if event_vars_to_add:
                data_logger.info(f"Adding event variables for future periods: {event_vars_to_add}")
                
                # Initialize with zeros
                for var in event_vars_to_add:
                    future_exog[var] = 0
                    
                # Look for repeating patterns in holidays/events (e.g., weekends, holidays)
                for var in event_vars_to_add:
                    if var == 'holiday':
                        # Mark weekends as holidays if they were marked as such in historical data
                        weekend_holiday_count = df[(df['is_weekend'] == 1) & (df['holiday'] == 1)].shape[0]
                        weekend_count = df[df['is_weekend'] == 1].shape[0]
                        
                        if weekend_holiday_count > 0.5 * weekend_count:  # If more than half of weekends are holidays
                            future_exog.loc[future_exog['ds'].dt.dayofweek.isin([5, 6]), 'holiday'] = 1
                    
            # For other exogenous variables, use mean values or other appropriate methods
            for var in exogenous_variables:
                if var not in future_exog.columns and var in df.columns:
                    if np.issubdtype(df[var].dtype, np.number):
                        # For numeric variables, use mean of recent values
                        future_exog[var] = df[var].tail(recent_days).mean()
                    else:
                        # For categorical variables, use most common value
                        future_exog[var] = df[var].mode()[0]
            
            return future_exog
            
        except Exception as e:
            data_logger.error(f"Error preparing future exogenous dataframe: {str(e)}")
            data_logger.error(traceback.format_exc())
            # Return a basic dataframe with just dates as fallback
            return pd.DataFrame({'ds': future_dates})
        
        



    '''
    def generate_forecast(self, 
                        model_id: str, 
                        df: pd.DataFrame,
                        freq: str = 'D',
                        periods: int = 30,
                        include_weather: bool = True,
                        include_events: bool = True,
                        include_time_features: bool = True,
                        force_retrain: bool = False,
                        finetune: bool = True,
                        finetune_steps: int = 10,
                        finetune_loss: str = "mse",
                        finetune_depth: int = 2) -> Dict[str, Any]:
        """
        Generate forecast using a saved model or train a new one
        
        Args:
            model_id: Unique identifier for the model
            df: Input DataFrame in TimeGPT format
            periods: Number of periods to forecast
            include_weather: Whether to include weather as regressors
            include_events: Whether to include events as regressors
            include_time_features: Whether to include time features
            force_retrain: Whether to force model retraining
            finetune: Whether to fine-tune the model
            finetune_steps: Number of fine-tuning steps
            finetune_loss: Loss function for fine-tuning
            finetune_depth: Level of fine-tuning (1-5)
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Check for existing model metadata
            metadata_exists = os.path.exists(os.path.join(self.model_dir, f"{model_id}_metadata.json"))
            
            if metadata_exists and not force_retrain:
                # Load existing model metadata
                model_metadata = self.load_model_metadata(model_id)
                
                if model_metadata is None:
                    data_logger.warning(f"Failed to load model metadata for {model_id}, retraining")
                    model_metadata, forecast = self.train_model(
                        df, 
                        model_id, 
                        freq,
                        periods,
                        include_weather, 
                        include_events,
                        include_time_features,
                        finetune,
                        finetune_steps,
                        finetune_loss,
                        finetune_depth
                    )
                else:
                    data_logger.info(f"Using existing TimeGPT model: {model_id}")
                    
                    # Get exogenous variables from metadata
                    exogenous_variables = model_metadata.get('exogenous_variables', [])
                    
                    # Prepare future exogenous dataframe
                    future_exog = self._prepare_future_exog_df(df, periods, exogenous_variables)
                    
                    # Generate forecast parameters
                    forecast_params = {
                        'df': df,
                        'h': periods,
                        'level': [80, 90],  # Confidence intervals
                        'time_col': 'ds',
                        'target_col': 'y',
                        'model': self.model_type
                    }
                    
                    # Add exogenous variables if any
                    if exogenous_variables and not future_exog.empty:
                        forecast_params['X_df'] = future_exog
                    
                    # Add fine-tuned model ID if available
                    finetuned_model_id = model_metadata.get('finetuned_model_id')
                    if finetuned_model_id:
                        forecast_params['finetuned_model_id'] = finetuned_model_id
                    
                    # Generate forecast
                    forecast = self.client.forecast(**forecast_params)
            else:
                # Train new model
                data_logger.info(f"Training new model (model not found or force_retrain={force_retrain})")
                model_metadata, forecast = self.train_model(
                    df, 
                    model_id, 
                    freq,
                    periods,
                    include_weather, 
                    include_events,
                    include_time_features,
                    finetune,
                    finetune_steps,
                    finetune_loss,
                    finetune_depth
                )
            
            if model_metadata is None:
                return {
                    "success": False,
                    "error": "Failed to create or load model"
                }
            
            # Process forecast results to match Prophet's format
            forecast_df = forecast.copy()
            
            # Rename columns to match Prophet's format
            forecast_df = forecast_df.rename(columns={
                'TimeGPT': 'yhat',
                'TimeGPT-lo-80': 'yhat_lower',
                'TimeGPT-hi-80': 'yhat_upper'
            })
            
            # Merge with original data to include actuals
            result = pd.merge(
                forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
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
            
            # Convert to lists for JSON serialization
            dates = result['ds'].dt.strftime('%Y-%m-%d').tolist()
            actuals = result['y'].tolist()
            predictions = result['yhat'].tolist()
            lower_bound = result['yhat_lower'].tolist()
            upper_bound = result['yhat_upper'].tolist()
            
            # Create forecast response similar to Prophet
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
                "model_info": {
                    "model_type": self.model_type,
                    "fine_tuned": finetune,
                    "finetuned_model_id": model_metadata.get('finetuned_model_id')
                }
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
                        freq: str = 'D',
                        periods: int = 30,
                        include_weather: bool = True,
                        include_events: bool = True,
                        include_time_features: bool = True,
                        force_retrain: bool = False,
                        finetune: bool = True,
                        finetune_steps: int = 10,
                        finetune_loss: str = "mse",
                        finetune_depth: int = 2) -> Dict[str, Any]:
        """
        Generate forecast using a saved model or train a new one,
        including both historical and future predictions (similar to Prophet).
        
        Args:
            model_id: Unique identifier for the model
            df: Input DataFrame in TimeGPT format
            periods: Number of periods to forecast
            include_weather: Whether to include weather as regressors
            include_events: Whether to include events as regressors
            include_time_features: Whether to include time features
            force_retrain: Whether to force model retraining
            finetune: Whether to fine-tune the model
            finetune_steps: Number of fine-tuning steps
            finetune_loss: Loss function for fine-tuning
            finetune_depth: Level of fine-tuning (1-5)
            
        Returns:
            Dictionary with forecast results for both historical and future periods
        """
        try:
            # Explicitly ensure consistent date formatting and handle missing dates
            df_copy = df.copy()
            df_copy['ds'] = pd.to_datetime(df_copy['ds'])
            
            # Sort by date
            df_copy = df_copy.sort_values('ds')
            
            # Check for and remove duplicate timestamps
            if df_copy['ds'].duplicated().any():
                duplicates = df_copy['ds'].duplicated().sum()
                df_copy = df_copy.drop_duplicates('ds')
                data_logger.warning(f"Dropped {duplicates} duplicate timestamps")
                
            # Check for missing dates in the sequence
            date_range = pd.date_range(start=df_copy['ds'].min(), end=df_copy['ds'].max(), freq=freq)
            missing_dates = set(date_range) - set(df_copy['ds'])
            
            if missing_dates:
                data_logger.warning(f"Found {len(missing_dates)} missing dates in the time series")
                
                # Create DataFrame with missing dates
                missing_df = pd.DataFrame({'ds': list(missing_dates)})
                
                # Fill with median values for numeric columns
                for col in df_copy.columns:
                    if col != 'ds':
                        if pd.api.types.is_numeric_dtype(df_copy[col]):
                            missing_df[col] = df_copy[col].median()
                        else:
                            # For categorical columns, use mode
                            missing_df[col] = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 0
                
                # Combine with original data and sort
                df_copy = pd.concat([df_copy, missing_df]).sort_values('ds')
                data_logger.info(f"Filled {len(missing_dates)} missing dates with median/mode values")
            
            # Check for existing model metadata
            metadata_exists = os.path.exists(os.path.join(self.model_dir, f"{model_id}_metadata.json"))
            
            if metadata_exists and not force_retrain:
                # Load existing model metadata
                model_metadata = self.load_model_metadata(model_id)
                
                if model_metadata is None:
                    data_logger.warning(f"Failed to load model metadata for {model_id}, retraining")
                    model_metadata, _ = self.train_model(
                        df_copy, 
                        model_id, 
                        freq,
                        periods,
                        include_weather, 
                        include_events,
                        include_time_features,
                        finetune,
                        finetune_steps,
                        finetune_loss,
                        finetune_depth
                    )
            else:
                # Train new model
                data_logger.info(f"Training new model (model not found or force_retrain={force_retrain})")
                model_metadata, _ = self.train_model(
                    df_copy, 
                    model_id, 
                    freq,
                    periods,
                    include_weather, 
                    include_events,
                    include_time_features,
                    finetune,
                    finetune_steps,
                    finetune_loss,
                    finetune_depth
                )
            
            if model_metadata is None:
                return {
                    "success": False,
                    "error": "Failed to create or load model"
                }
            
            # Get exogenous variables from metadata
            exogenous_variables = model_metadata.get('exogenous_variables', [])
            
            # Get fine-tuned model ID
            finetuned_model_id = model_metadata.get('finetuned_model_id')
            
            try:
                # Prepare future exogenous dataframe for future periods
                future_exog = self._prepare_future_exog_df(df_copy, periods, exogenous_variables)
                
                # Generate forecast parameters with explicit frequency and add_history=True
                forecast_params = {
                    'df': df_copy,
                    'h': periods,
                    'freq': freq,
                    'level': [80, 90],  # Confidence intervals
                    'time_col': 'ds',
                    'target_col': 'y',
                    'model': self.model_type,
                    'add_history': True  # Add this parameter to get historical predictions
                }
                
                # Add future exogenous variables if any
                if exogenous_variables and not future_exog.empty:
                    forecast_params['X_df'] = future_exog
                
                # Add fine-tuned model ID if available
                if finetuned_model_id:
                    forecast_params['finetuned_model_id'] = finetuned_model_id
                
                # Generate forecast with historical predictions
                data_logger.info(f"Generating forecast with historical predictions using fine-tuned model: {finetuned_model_id}")
                forecast_result = self.client.forecast(**forecast_params)
                
                # Rename columns to match Prophet's format
                forecast_result = forecast_result.rename(columns={
                    'TimeGPT': 'yhat',
                    'TimeGPT-lo-80': 'yhat_lower',
                    'TimeGPT-hi-80': 'yhat_upper'
                })
                
                # Since the API now returns both historical and future predictions,
                # we can merge with original data to include actuals
                result = pd.merge(
                    forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                    df_copy[['ds', 'y']],
                    on='ds',
                    how='outer'
                )
                
                # Sort by date
                result = result.sort_values('ds')
                
                # Calculate accuracy metrics for historical period
                historical = result.dropna(subset=['y', 'yhat'])
                
                if len(historical) > 0:
                    # Calculate MAPE
                    mape = np.mean(np.abs((historical['y'] - historical['yhat']) / historical['y'])) * 100
                    
                    # Calculate RMSE
                    rmse = np.sqrt(np.mean((historical['y'] - historical['yhat']) ** 2))
                    
                    data_logger.info(f"Calculated metrics on {len(historical)} historical points: MAPE={mape:.2f}%, RMSE={rmse:.2f}")
                else:
                    mape = None
                    rmse = None
                    data_logger.warning(f"No historical data available for evaluation")
                
                # Convert to lists for JSON serialization
                dates = result['ds'].dt.strftime('%Y-%m-%d').tolist()
                actuals = result['y'].tolist()
                predictions = result['yhat'].tolist()
                lower_bound = result['yhat_lower'].tolist()
                upper_bound = result['yhat_upper'].tolist()
                
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
                    "model_info": {
                        "model_type": self.model_type,
                        "fine_tuned": finetune,
                        "finetuned_model_id": model_metadata.get('finetuned_model_id')
                    }
                }
                
                # Add metadata
                response["metadata"] = {
                    "company": model_id.split('_')[0] if '_' in model_id else "",
                    "target": model_id.split('_')[1] if len(model_id.split('_')) > 1 else "unknown",
                    "model_type": "timegpt",
                    "generated_at": datetime.now().isoformat(),
                    "periods": periods,
                    "include_weather": include_weather,
                    "include_events": include_events,
                    "include_time_features": include_time_features,
                    "force_retrain": force_retrain,
                    "data_range": {
                        "start": df_copy['ds'].min().strftime('%Y-%m-%d'),
                        "end": df_copy['ds'].max().strftime('%Y-%m-%d')
                    },
                    "timegpt_params": {
                        "finetune": finetune,
                        "finetune_steps": finetune_steps,
                        "finetune_loss": finetune_loss,
                        "finetune_depth": finetune_depth
                    },
                    "with_history": True
                }
                
                data_logger.info(f"Generated forecast for model: {model_id} with historical predictions")
                return response
                
            except Exception as e:
                data_logger.error(f"Error generating forecast: {str(e)}")
                data_logger.error(traceback.format_exc())
                return {
                    "success": False,
                    "error": str(e)
                }
                
        except Exception as e:
            data_logger.error(f"Error in generate_forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
        



    def cross_validate(self, 
                      df: pd.DataFrame, 
                      model_id: str,
                      h: int = 7,
                      n_windows: int = 3,
                      include_weather: bool = True,
                      include_events: bool = True,
                      include_time_features: bool = True,
                      finetune: bool = True,
                      finetune_steps: int = 10) -> Dict[str, Any]:
        """
        Perform cross-validation for the TimeGPT model
        
        Args:
            df: Input DataFrame in TimeGPT format
            model_id: Unique identifier for the model
            h: Forecast horizon
            n_windows: Number of test windows
            include_weather: Whether to include weather as regressors
            include_events: Whether to include events as regressors
            include_time_features: Whether to include time features
            finetune: Whether to fine-tune the model
            finetune_steps: Number of fine-tuning steps
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            data_logger.info(f"Performing {n_windows}-window cross-validation with horizon {h}")
            
            # Get exogenous variables
            exogenous_variables = self._get_exogenous_variables(
                df, include_weather, include_events, include_time_features
            )
            
            # Set up cross validation parameters
            cv_params = {
                'df': df,
                'h': h,
                'n_windows': n_windows,
                'time_col': 'ds',
                'target_col': 'y',
                'model': self.model_type
            }
            
            # Add exogenous variables if any
            if exogenous_variables:
                cv_params['hist_exog_list'] = exogenous_variables
            
            # Add fine-tuning parameters if enabled
            if finetune:
                cv_params['finetune_steps'] = finetune_steps
            
            # Perform cross-validation
            cv_results = self.client.cross_validation(**cv_params)
            
            # Process cross-validation results
            # Calculate overall metrics
            historical_data = cv_results.copy()
            metrics = {}
            
            if len(historical_data) > 0:
                # Calculate MAPE
                actuals = historical_data['y'].values
                predictions = historical_data['TimeGPT'].values
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
            
            # Calculate metrics by cutoff
            cutoffs = historical_data['cutoff'].unique()
            cutoff_metrics = {}
            
            for cutoff in cutoffs:
                cutoff_data = historical_data[historical_data['cutoff'] == cutoff]
                
                if len(cutoff_data) > 0:
                    # Calculate MAPE for this cutoff
                    actuals = cutoff_data['y'].values
                    predictions = cutoff_data['TimeGPT'].values
                    abs_diff = np.abs(actuals - predictions)
                    cutoff_mape = np.mean(abs_diff / np.abs(actuals)) * 100 if np.sum(np.abs(actuals)) > 0 else None
                    
                    # Calculate RMSE for this cutoff
                    cutoff_rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
                    
                    cutoff_metrics[str(cutoff)] = {
                        'mape': cutoff_mape,
                        'rmse': cutoff_rmse,
                        'n_points': len(cutoff_data)
                    }
            
            # Create response
            response = {
                "success": True,
                "model_id": model_id,
                "overall_metrics": metrics,
                "cutoff_metrics": cutoff_metrics,
                "cv_config": {
                    "horizon": h,
                    "windows": n_windows,
                    "exogenous_variables": exogenous_variables,
                    "finetune": finetune,
                    "finetune_steps": finetune_steps
                }
            }
            
            data_logger.info(f"Cross-validation completed successfully for model: {model_id}")
            return response
            
        except Exception as e:
            data_logger.error(f"Error performing cross-validation: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

def test_timegpt_model():
    """Test the TimeGPTModel functionality with sample data"""
    import os
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Running TimeGPT model test...")
    
    try:
        # Create sample time series data for testing
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducibility
        
        # Create sample daily sales data with trend and seasonality
        sales = np.random.normal(loc=1000, scale=50, size=len(dates))
        # Add trend
        sales = sales + np.arange(len(dates)) * 0.5
        # Add weekly seasonality
        weekday_effect = 100 * np.sin(np.arange(len(dates)) * (2 * np.pi / 7))
        # Add monthly seasonality
        monthly_effect = 200 * np.sin(np.arange(len(dates)) * (2 * np.pi / 30))
        
        sales = sales + weekday_effect + monthly_effect
        
        # Create dataframe
        df = pd.DataFrame({
            'date': dates,
            'total_revenue': sales,
            'temperature': 20 + 15 * np.sin(np.arange(len(dates)) * (2 * np.pi / 365)),
            'precipitation': np.random.uniform(0, 10, size=len(dates))
        })
        
        # Add categorical features
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        df['is_holiday'] = 0
        df.loc[df['date'].isin(['2023-01-01', '2023-12-25']), 'is_holiday'] = 1
        
        print(f"Created sample data with {len(df)} rows")
        
        # Initialize TimeGPT model with environment variable
        # Note: For actual testing, set the TIMEGPT_API_KEY environment variable
        # or pass the API key directly to the TimeGPTModel constructor
        api_key = os.environ.get('TIMEGPT_API_KEY')
        if api_key:
            print("Using API key from environment variable")
            model = TimeGPTModel(api_key=api_key)
        else:
            print("No API key found, skipping test.")
            return "Test skipped: No API key available"
        
        # Create TimeGPT dataframe
        timegpt_df = model.prepare_data_for_timegpt(df, "total_revenue")
        print(f"Prepared TimeGPT dataframe with {len(timegpt_df)} rows")
        
        # Add time features
        timegpt_df = model.add_time_features(timegpt_df)
        print("Added time features")
        
        # Create sample events dataframe
        event_dates = ['2023-01-01', '2023-07-04', '2023-12-25']
        events_df = pd.DataFrame({
            'ds': pd.to_datetime(event_dates),
            'event': 1,
            'holiday': 1,
            'festival': 0
        })
        
        # Add events features
        timegpt_df = model.add_events_features(timegpt_df, events_df)
        print("Added events features")
        
        # Create sample weather dataframe
        weather_df = df[['date', 'temperature', 'precipitation']].copy()
        weather_df = weather_df.rename(columns={'date': 'ds'})
        
        # Add weather features
        timegpt_df = model.add_weather_features(timegpt_df, weather_df)
        print("Added weather features")
        
        # Create test directory if it doesn't exist
        os.makedirs("data/test", exist_ok=True)
        
        # Save TimeGPT dataframe for reference
        timegpt_df.to_csv("data/test/test_timegpt_data.csv", index=False)
        print("Saved TimeGPT test dataframe")
        
        # Test model training
        model_id = "test_revenue_timegpt"
        print(f"Training model: {model_id}")
        freq = 'D'
        model_metadata, forecast = model.train_model(
            timegpt_df,
            model_id,
            freq,
            periods=7,
            include_weather=True,
            include_events=True,
            include_time_features=True,
            finetune=True,
            finetune_steps=5,
            finetune_loss="mse",
            finetune_depth=1
        )
        
        if model_metadata:
            print(f"Model trained successfully: {model_id}")
            print(f"Fine-tuned model ID: {model_metadata.get('finetuned_model_id')}")
        else:
            print("Model training failed")
            return "Test failed: Model training error"
        
        # Test forecast generation
        print("Generating forecast")
        forecast_result = model.generate_forecast(
            model_id,
            timegpt_df,
            periods=7,
            include_weather=True,
            include_events=True,
            include_time_features=True,
            force_retrain=False
        )
        
        if forecast_result.get('success'):
            print("Forecast generated successfully")
            print(f"Forecast periods: {len(forecast_result['dates']) - len(timegpt_df)}")
            if forecast_result['metrics'].get('mape') is not None:
                print(f"MAPE: {forecast_result['metrics']['mape']:.2f}%")
        else:
            print(f"Forecast generation failed: {forecast_result.get('error')}")
            return "Test failed: Forecast generation error"
        
        # Test cross-validation
        print("Performing cross-validation")
        cv_result = model.cross_validate(
            timegpt_df,
            model_id,
            h=7,
            n_windows=2,
            include_weather=True,
            include_events=True,
            include_time_features=True,
            finetune=True,
            finetune_steps=5
        )
        
        if cv_result.get('success'):
            print("Cross-validation completed successfully")
            if cv_result['overall_metrics'].get('mape') is not None:
                print(f"CV MAPE: {cv_result['overall_metrics']['mape']:.2f}%")
        else:
            print(f"Cross-validation failed: {cv_result.get('error')}")
        
        print("TimeGPT model test completed successfully")
        return "TimeGPT model test completed successfully"
        
    except Exception as e:
        import traceback
        print(f"Error in test_timegpt_model: {str(e)}")
        print(traceback.format_exc())
        return f"Test failed: {str(e)}"

if __name__ == "__main__":
    test_timegpt_model()