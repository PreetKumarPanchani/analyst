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
import itertools

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.logger import data_logger

class ProphetModel:
    """
    Prophet model for multivariate time series forecasting with external regressors
    """
    
    def __init__(self, model_dir: str = "data/models"):
        """Initialize the Prophet model"""
        self.model_dir = model_dir
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
        

    def _get_cv_params_for_dataset(self, df: pd.DataFrame, requested_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate appropriate cross-validation parameters based on dataset size
        
        Args:
            df: Input DataFrame in Prophet format
            requested_params: Requested CV parameters (will be adjusted if needed)
            
        Returns:
            Dictionary with safe CV parameters
        """
        # Get dataset date range
        min_date = df['ds'].min()
        max_date = df['ds'].max()
        date_range = (max_date - min_date).days
        
        # Default parameters if none provided
        if requested_params is None:
            requested_params = {
                'initial': '180 days',
                'period': '30 days',
                'horizon': '30 days',
                'parallel': 'processes'
            }
        
        # Parse the requested parameters
        def parse_days(param):
            if isinstance(param, str) and 'days' in param:
                return int(param.split(' ')[0])
            elif isinstance(param, str) and 'day' in param:
                return int(param.split(' ')[0])
            elif isinstance(param, int):
                return param
            else:
                return 30  # Default
        
        initial_days = parse_days(requested_params.get('initial', '180 days'))
        horizon_days = parse_days(requested_params.get('horizon', '30 days'))
        period_days = parse_days(requested_params.get('period', '30 days'))
        
        # Calculate minimum required days
        min_required_days = initial_days + horizon_days
        
        # Check if we have enough data
        if date_range < min_required_days:
            data_logger.warning(f"Dataset too small for requested CV parameters. Adjusting parameters.")
            
            # Adjust parameters to fit the dataset
            if date_range < 90:
                # Too little data for meaningful CV
                data_logger.warning(f"Dataset with {date_range} days is too small for meaningful cross-validation")
                return None
            
            # Allocate 70% for initial and rest for horizon
            initial_days = int(date_range * 0.7)
            horizon_days = min(30, date_range - initial_days - 1)
            period_days = min(period_days, horizon_days)
        
        # Create safe parameters
        safe_params = {
            'initial': f'{initial_days} days',
            'horizon': f'{horizon_days} days',
            'period': f'{period_days} days',
            'parallel': requested_params.get('parallel', 'processes')
        }
        
        data_logger.info(f"Using CV parameters: {safe_params}")
        return safe_params

    def train_model(self, 
                    df: pd.DataFrame, 
                    model_id: str,
                    periods: int = 30,
                    include_weather: bool = True,
                    include_events: bool = True,
                    include_time_features: bool = True,
                    perform_cv: bool = False,
                    cv_params: Dict[str, Any] = None,
                    tune_hyperparams: bool = False,
                    param_grid: Dict[str, List] = None) -> Tuple[Prophet, pd.DataFrame]:
        """
        Train a Prophet model with regressors, optionally performing cross-validation and hyperparameter tuning
        
        Args:
            df: Input DataFrame in Prophet format (with ds, y)
            model_id: Unique identifier for the model
            periods: Number of periods to forecast
            include_weather: Whether to include weather as regressors
            include_events: Whether to include events as regressors
            include_time_features: Whether to include time-based features as regressors
            perform_cv: Whether to perform cross-validation
            cv_params: Cross-validation parameters
            tune_hyperparams: Whether to tune hyperparameters
            param_grid: Grid of hyperparameters to search
            
        Returns:
            Tuple of (Prophet model, forecast DataFrame)
        """
        try:
            data_logger.info(f"Training Prophet model: {model_id}")
            model_params = {}
            
            # Check if dataset is suitable for cross-validation
            if (perform_cv or tune_hyperparams) and len(df) > 0:
                # Get safe CV parameters for this dataset
                safe_cv_params = self._get_cv_params_for_dataset(df, cv_params)
                
                if safe_cv_params is None:
                    data_logger.warning(f"Dataset too small for cross-validation. Skipping CV and hyperparameter tuning.")
                    perform_cv = False
                    tune_hyperparams = False
                else:
                    cv_params = safe_cv_params
            
            # Tune hyperparameters if requested
            if tune_hyperparams:
                if param_grid is None:
                    # Default parameter grid if none provided - using a smaller grid for efficiency
                    param_grid = {
                        'changepoint_prior_scale': [0.01, 0.1, 0.5],
                        'seasonality_prior_scale': [0.1, 1.0, 10.0],
                        'seasonality_mode': ['additive', 'multiplicative'],
                        'daily_seasonality': [False, True],
                    }
                
                data_logger.info(f"Tuning hyperparameters for model: {model_id}")
                
                # Generate and test parameter combinations
                best_params = {}
                best_rmse = float('inf')
                
                # Generate all combinations of parameters
                keys = param_grid.keys()
                values = param_grid.values()
                param_combinations = list(itertools.product(*values))
                
                for values in param_combinations:
                    params = dict(zip(keys, values))
                    # Create model with these parameters
                    m = Prophet(**params)
                    
                    # Add same regressors as we'll use in the final model
                    if include_weather:
                        if 'temperature' in df.columns:
                            m.add_regressor('temperature')
                        if 'precipitation' in df.columns:
                            m.add_regressor('precipitation')
                        if 'rainy' in df.columns:
                            m.add_regressor('rainy')
                        if 'sunny' in df.columns:
                            m.add_regressor('sunny')
                    
                    if include_events:
                        if 'holiday' in df.columns:
                            m.add_regressor('holiday')
                        if 'festival' in df.columns:
                            m.add_regressor('festival')
                        if 'event' in df.columns:
                            m.add_regressor('event')

                    if include_time_features:
                        m.add_regressor('dayofweek', standardize=False)
                        m.add_regressor('is_weekend', standardize=False)
                        m.add_regressor('month', standardize=False)
                        m.add_regressor('year', standardize=False)
                        
                    # Fit the model
                    m.fit(df)
                    
                    # Perform cross validation
                    try:
                        from prophet.diagnostics import cross_validation, performance_metrics
                        
                        cv_results = cross_validation(
                            model=m,
                            initial=cv_params['initial'],
                            period=cv_params['period'],
                            horizon=cv_params['horizon'],
                            parallel=cv_params.get('parallel', 'processes')
                        )
                        
                        # Calculate performance metrics
                        cv_metrics = performance_metrics(cv_results)
                        
                        # Get RMSE
                        rmse = cv_metrics['rmse'].mean()
                        
                        # Update best parameters if this is better
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = params.copy()
                            data_logger.info(f"New best parameters found: {best_params} with RMSE={best_rmse:.4f}")
                    except Exception as e:
                        data_logger.error(f"Error in hyperparameter CV with params {params}: {str(e)}")
                        # Continue with next parameter set instead of failing completely
                        continue
                
                if best_params:
                    data_logger.info(f"Best parameters found: {best_params} with RMSE={best_rmse:.4f}")
                    model_params = best_params
                else:
                    data_logger.warning("Hyperparameter tuning failed to find optimal parameters, using defaults")
            
            # Initialize Prophet model with tuned parameters or defaults
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=model_params.get('daily_seasonality', False),
                seasonality_mode=model_params.get('seasonality_mode', 'multiplicative'),
                changepoint_prior_scale=model_params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=model_params.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=model_params.get('holidays_prior_scale', 10.0)
            )
            
            # Add regressors if present
            if include_weather:
                if 'temperature' in df.columns:
                    model.add_regressor('temperature')
                if 'precipitation' in df.columns:
                    model.add_regressor('precipitation')
                if 'rainy' in df.columns:
                    model.add_regressor('rainy')
                if 'sunny' in df.columns:
                    model.add_regressor('sunny')
            
            if include_events:
                if 'holiday' in df.columns:
                    model.add_regressor('holiday')
                if 'festival' in df.columns:
                    model.add_regressor('festival')
                if 'event' in df.columns:
                    model.add_regressor('event')

            # Add time-based features as regressors if requested
            if include_time_features:
                # Ensure time features exist
                df = self.add_time_features(df)
                
                # Add time-based regressors
                model.add_regressor('dayofweek', standardize=False)
                model.add_regressor('is_weekend', standardize=False)
                model.add_regressor('month', standardize=False)
                model.add_regressor('year', standardize=False)
            
            # Fit the model
            model.fit(df)
            
            # Perform cross-validation if requested (and wasn't already done for hyperparameter tuning)
            if perform_cv and not tune_hyperparams:
                from prophet.diagnostics import cross_validation, performance_metrics
                
                try:
                    # Perform cross validation
                    cv_results = cross_validation(
                        model=model,
                        initial=cv_params['initial'],
                        period=cv_params['period'],
                        horizon=cv_params['horizon'],
                        parallel=cv_params.get('parallel', 'processes')
                    )
                    
                    # Calculate performance metrics
                    cv_metrics = performance_metrics(cv_results)
                    
                    # Log metrics
                    data_logger.info(f"Cross-validation metrics for {model_id}:")
                    data_logger.info(f"RMSE: {cv_metrics['rmse'].mean():.4f}")
                    data_logger.info(f"MAPE: {cv_metrics['mape'].mean():.4f}%")
                    data_logger.info(f"MAE: {cv_metrics['mae'].mean():.4f}")
                    
                    # Save CV results to file
                    cv_dir = os.path.join(self.model_dir, "cv_results")
                    os.makedirs(cv_dir, exist_ok=True)
                    cv_path = os.path.join(cv_dir, f"{model_id}_cv_metrics.json")
                    
                    # Convert metrics to serializable format
                    metrics_dict = {
                        'rmse': float(cv_metrics['rmse'].mean()),
                        'mape': float(cv_metrics['mape'].mean()),
                        'mae': float(cv_metrics['mae'].mean()),
                        'horizon': cv_params['horizon'],
                        'initial': cv_params['initial'],
                        'period': cv_params['period']
                    }
                    
                    with open(cv_path, 'w') as f:
                        json.dump(metrics_dict, f)
                        
                    data_logger.info(f"Saved cross-validation metrics to {cv_path}")
                    
                except Exception as e:
                    data_logger.error(f"Error performing cross-validation: {str(e)}")
                    data_logger.error(traceback.format_exc())
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)
            
            # Add time-based features to future
            if include_time_features:
                future = self.add_time_features(future)
            
            # Add regressor values to future
            for col in df.columns:
                if col not in ['ds', 'y', 'dayofweek', 'is_weekend', 'month', 'year']:
                    # Copy last value for forecast period
                    if df[col].dtype == 'category':
                        future[col] = pd.Categorical([0] * len(future), categories=df[col].cat.categories)
                    else:
                        future[col] = 0
                    
                    # Copy known values
                    for idx, row in df.iterrows():
                        mask = future['ds'] == row['ds']
                        if any(mask):
                            future.loc[mask, col] = row[col]
            
            # Make forecast
            forecast = model.predict(future)
            
            # Save model - Always save the model for future use
            self._save_model(model, model_id)
            
            data_logger.info(f"Prophet model trained successfully: {model_id}")
            return model, forecast
            
        except Exception as e:
            data_logger.error(f"Error training Prophet model: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None, pd.DataFrame()
        
        
    def load_model(self, model_id: str) -> Optional[Prophet]:
        """
        Load a saved Prophet model
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Prophet model (or None if not found)
        """
        try:
            model_path = os.path.join(self.model_dir, f"{model_id}.json")
            
            if not os.path.exists(model_path):
                data_logger.warning(f"Model not found: {model_path}")
                return None
            
            with open(model_path, 'r') as f:
                model_json = json.load(f)
                
            model = model_from_json(model_json)
            
            data_logger.info(f"Successfully loaded saved model: {model_id}")
            return model
            
        except Exception as e:
            data_logger.error(f"Error loading model {model_id}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None
    
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
            # Create model directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            model_path = os.path.join(self.model_dir, f"{model_id}.json")
            
            # Serialize and save the model
            with open(model_path, 'w') as f:
                json.dump(model_to_json(model), f)
                
            data_logger.info(f"Saved model weights to: {model_path}")
            return True
            
        except Exception as e:
            data_logger.error(f"Error saving model {model_id}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def generate_forecast(self, 
                         model_id: str, 
                         df: pd.DataFrame, 
                         periods: int = 30,
                         include_weather: bool = True,
                         include_events: bool = True,
                         include_time_features: bool = True,
                         force_retrain: bool = False,
                         perform_cv: bool = False,
                         cv_params: Dict[str, Any] = None,
                         tune_hyperparams: bool = False,
                         param_grid: Dict[str, List] = None) -> Dict[str, Any]:
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
            perform_cv: Whether to perform cross-validation
            cv_params: Cross-validation parameters
            tune_hyperparams: Whether to tune hyperparameters
            param_grid: Grid of hyperparameters to search
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Try to load existing model (unless force_retrain is True)
            model = None if force_retrain else self.load_model(model_id)
            
            # Train new model if not found or force_retrain is True
            if model is None:
                data_logger.info(f"Training new model (model not found or force_retrain={force_retrain})")
                model, forecast = self.train_model(
                    df, 
                    model_id, 
                    periods, 
                    include_weather, 
                    include_events,
                    include_time_features,
                    perform_cv=perform_cv,
                    cv_params=cv_params,
                    tune_hyperparams=tune_hyperparams,
                    param_grid=param_grid
                )
            else:
                data_logger.info(f"Using existing model: {model_id}")
                # Create future dataframe
                future = model.make_future_dataframe(periods=periods)
                
                # Add time-based features to future
                if include_time_features:
                    future = self.add_time_features(future)
                
                # Add regressor values to future
                for col in df.columns:
                    if col not in ['ds', 'y', 'dayofweek', 'is_weekend' ,'month', 'year']:
                        # Copy last value for forecast period
                        if df[col].dtype == 'category':
                            future[col] = pd.Categorical([0] * len(future), categories=df[col].cat.categories)
                        else:
                            future[col] = 0
                        
                        # Copy known values
                        for idx, row in df.iterrows():
                            mask = future['ds'] == row['ds']
                            if any(mask):
                                future.loc[mask, col] = row[col]
                
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
                "components": {
                    "trend": forecast['trend'].tolist(),
                    "weekly": forecast['weekly'].tolist() if 'weekly' in forecast.columns else None,
                    "yearly": forecast['yearly'].tolist() if 'yearly' in forecast.columns else None
                }
            }
            
            data_logger.info(f"Generated forecast for model: {model_id}")
            
            def ensure_serializable(obj):
                """Convert complex objects to serializable types"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: ensure_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [ensure_serializable(i) for i in obj]
                return obj

            response = ensure_serializable(response)
            
            return response
            
        except Exception as e:
            data_logger.error(f"Error generating forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

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

        print(f"Prophet dataframe: {prophet_df.head()}")
        
        # Create directory if it doesn't exist
        data_dir = os.path.join("data", company, "revenue")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save prophet dataframe
        prophet_df.to_csv(os.path.join(data_dir, f"{company}_prophet_revenue_data.csv"), index=False)
        print(f"Prophet dataframe saved to: {os.path.join(data_dir, f'{company}_prophet_revenue_data.csv')}")

        # Test model training with time features
        cv_params={
            'initial': '180 days',
            'period': '30 days',
            'horizon': '30 days',
        }
        model_id = f"{company}_revenue"
        print(f"Training model: {model_id}")
        _, forecast = model.train_model(
            prophet_df, 
            model_id, 
            periods=5,
            include_weather=True,
            include_events=True,
            include_time_features=True,
            perform_cv=True,
            cv_params=cv_params,
            tune_hyperparams=True,
            param_grid=None
        )
        
        # Test model loading
        print(f"Testing model loading: {model_id}")
        loaded_model = model.load_model(model_id)
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
            force_retrain=False,
            perform_cv=True,
            cv_params=cv_params,
            tune_hyperparams=True,
            param_grid=None

        )
        
        print(f"Forecast success: {forecast_result['success']}")
        if forecast_result['success'] and forecast_result['metrics']['mape'] is not None:
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
            force_retrain=True,
            perform_cv=True,
            cv_params=cv_params,
            tune_hyperparams=True,
            param_grid=None,
        )
        
        print(f"Retrained forecast success: {forecast_result_retrained['success']}")
        
        return "Prophet model test completed successfully"
    else:
        return "No daily sales data available for testing"

if __name__ == "__main__":
    test_prophet_model()