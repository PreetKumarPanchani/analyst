import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import traceback
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from app.core.logger import data_logger
from app.core.config import settings

class WeatherService:
    """
    Service for retrieving weather data and forecasts for Sheffield, UK
    Uses the Open-Meteo API for current, forecast, and historical weather data
    """
    
    def __init__(self, cache_dir: str = 'data/cache'):
        """Initialize the weather service"""
        # Base URLs for Open-Meteo APIs
        self.forecast_api_url = "https://api.open-meteo.com/v1/forecast"
        self.historical_api_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Sheffield coordinates
        self.lat = settings.LOCATION_COORDINATES["sheffield"]["lat"]
        self.lon = settings.LOCATION_COORDINATES["sheffield"]["lon"]
        
        # Cache settings
        self.cache_dir = cache_dir 
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "weather_cache.json")
        self.cache_expiry = settings.CACHE_EXPIRY  # seconds
        
        data_logger.info(f"WeatherService initialized for Sheffield: {self.lat}, {self.lon}")
    
    def _get_cached_data(self, key: str) -> Tuple[bool, Any]:
        """
        Get data from cache if available and not expired
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (is_valid, data)
        """
        try:
            if not os.path.exists(self.cache_file):
                return False, None
            
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            
            if key not in cache:
                return False, None
            
            cached_item = cache[key]
            cached_time = datetime.fromisoformat(cached_item["timestamp"])
            current_time = datetime.now()
            
            # Check if cache is expired
            if (current_time - cached_time).total_seconds() > self.cache_expiry:
                data_logger.info(f"Cached weather data for '{key}' is expired")
                return False, None
            
            data_logger.info(f"Using cached weather data for '{key}'")
            return True, cached_item["data"]
            
        except Exception as e:
            data_logger.error(f"Error reading from cache: {str(e)}")
            return False, None
    
    def _set_cached_data(self, key: str, data: Any) -> bool:
        """
        Save data to cache
        
        Args:
            key: Cache key
            data: Data to cache
            
        Returns:
            Success flag
        """
        try:
            # Read existing cache or create new one
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
            else:
                cache = {}
            
            # Update cache
            cache[key] = {
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            # Write updated cache
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            
            data_logger.info(f"Cached weather data for '{key}'")
            return True
            
        except Exception as e:
            data_logger.error(f"Error writing to cache: {str(e)}")
            return False
    
    def get_current_weather(self) -> Dict[str, Any]:
        """
        Get current weather data for Sheffield using Open-Meteo
        
        Returns:
            Dictionary with weather data
        """
        try:
            # Check cache first
            cache_key = "current_weather"
            is_valid, cached_data = self._get_cached_data(cache_key)
            
            if is_valid:
                return cached_data
            
            # Make API request to Open-Meteo
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "current": [
                    "temperature_2m", 
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "is_day",
                    "precipitation",
                    "rain",
                    "showers",
                    "snowfall",
                    "weather_code",
                    "cloud_cover",
                    "pressure_msl",
                    "surface_pressure",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "wind_gusts_10m"
                ]
            }
            
            response = requests.get(self.forecast_api_url, params=params)
            
            if response.status_code != 200:
                data_logger.error(f"Error from Open-Meteo API: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            current_data = data.get("current", {})
            
            # Map weather code to weather description
            weather_code = current_data.get("weather_code")
            weather_main, weather_description = self._get_weather_description(weather_code)
            
            # Simplify the response
            weather_data = {
                "timestamp": datetime.now().isoformat(),
                "temperature": current_data.get("temperature_2m"),
                "feels_like": current_data.get("apparent_temperature"),
                "humidity": current_data.get("relative_humidity_2m"),
                "pressure": current_data.get("surface_pressure"),
                "weather_main": weather_main,
                "weather_description": weather_description,
                "weather_code": weather_code,
                "wind_speed": current_data.get("wind_speed_10m"),
                "wind_direction": current_data.get("wind_direction_10m"),
                "wind_gusts": current_data.get("wind_gusts_10m"),
                "clouds": current_data.get("cloud_cover"),
                "is_day": current_data.get("is_day"),
                "precipitation": current_data.get("precipitation"),
                "rain": current_data.get("rain"),
                "snowfall": current_data.get("snowfall")
            }
            
            # Cache the result
            self._set_cached_data(cache_key, weather_data)
            
            data_logger.info(f"Retrieved current weather: {weather_data['temperature']}°C, {weather_data['weather_main']}")
            
            return weather_data
            
        except Exception as e:
            data_logger.error(f"Error getting current weather: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None
    
    def _get_weather_description(self, code: int) -> Tuple[str, str]:
        """
        Convert WMO weather code to description
        
        Args:
            code: WMO weather code
            
        Returns:
            Tuple of (main category, description)
        """
        # WMO Weather interpretation codes (WW)
        weather_codes = {
            0: ("Clear", "Clear sky"),
            1: ("Mainly Clear", "Mainly clear"),
            2: ("Partly Cloudy", "Partly cloudy"),
            3: ("Overcast", "Overcast"),
            45: ("Fog", "Fog"),
            48: ("Fog", "Depositing rime fog"),
            51: ("Drizzle", "Light drizzle"),
            53: ("Drizzle", "Moderate drizzle"),
            55: ("Drizzle", "Dense drizzle"),
            56: ("Freezing Drizzle", "Light freezing drizzle"),
            57: ("Freezing Drizzle", "Dense freezing drizzle"),
            61: ("Rain", "Slight rain"),
            63: ("Rain", "Moderate rain"),
            65: ("Rain", "Heavy rain"),
            66: ("Freezing Rain", "Light freezing rain"),
            67: ("Freezing Rain", "Heavy freezing rain"),
            71: ("Snow", "Slight snow fall"),
            73: ("Snow", "Moderate snow fall"),
            75: ("Snow", "Heavy snow fall"),
            77: ("Snow", "Snow grains"),
            80: ("Rain Showers", "Slight rain showers"),
            81: ("Rain Showers", "Moderate rain showers"),
            82: ("Rain Showers", "Violent rain showers"),
            85: ("Snow Showers", "Slight snow showers"),
            86: ("Snow Showers", "Heavy snow showers"),
            95: ("Thunderstorm", "Thunderstorm"),
            96: ("Thunderstorm", "Thunderstorm with slight hail"),
            99: ("Thunderstorm", "Thunderstorm with heavy hail")
        }
        
        if code in weather_codes:
            return weather_codes[code]
        return ("Unknown", "Unknown weather condition")
        
    def get_forecast(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get weather forecast for Sheffield using Open-Meteo
        
        Args:
            days: Number of days to forecast (max 16)
            
        Returns:
            List of dictionaries with daily forecast data
        """
        try:
            # Check cache first
            cache_key = f"forecast_{days}"
            is_valid, cached_data = self._get_cached_data(cache_key)
            
            if is_valid:
                return cached_data
            
            # Make API request to Open-Meteo
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "forecast_days": min(days, 16),  # Maximum 16 days
                "daily": [
                    "weather_code", 
                    "temperature_2m_max", 
                    "temperature_2m_min",
                    "apparent_temperature_max",
                    "apparent_temperature_min",
                    "precipitation_sum",
                    "rain_sum",
                    "showers_sum",
                    "snowfall_sum",
                    "precipitation_hours",
                    "precipitation_probability_max",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max",
                    "wind_direction_10m_dominant",
                    "shortwave_radiation_sum"
                ],
                "timezone": "GMT"  # Use GMT timezone for consistency
            }
            
            response = requests.get(self.forecast_api_url, params=params)
            
            if response.status_code != 200:
                data_logger.error(f"Error from Open-Meteo API: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            daily_data = data.get("daily", {})
            daily_time = daily_data.get("time", [])
            
            # Process the forecast data
            forecasts = []
            
            for i in range(len(daily_time)):
                date_str = daily_time[i]
                weather_code = daily_data.get("weather_code", [])[i] if "weather_code" in daily_data else None
                weather_main, _ = self._get_weather_description(weather_code) if weather_code is not None else ("Unknown", "Unknown")
                
                daily_forecast = {
                    "date": date_str,
                    "min_temp": daily_data.get("temperature_2m_min", [])[i] if "temperature_2m_min" in daily_data else None,
                    "max_temp": daily_data.get("temperature_2m_max", [])[i] if "temperature_2m_max" in daily_data else None,
                    "avg_temp": (daily_data.get("temperature_2m_min", [])[i] + daily_data.get("temperature_2m_max", [])[i]) / 2 if "temperature_2m_min" in daily_data and "temperature_2m_max" in daily_data else None,
                    "min_feels_like": daily_data.get("apparent_temperature_min", [])[i] if "apparent_temperature_min" in daily_data else None,
                    "max_feels_like": daily_data.get("apparent_temperature_max", [])[i] if "apparent_temperature_max" in daily_data else None,
                    "precipitation": daily_data.get("precipitation_sum", [])[i] if "precipitation_sum" in daily_data else None,
                    "rain": daily_data.get("rain_sum", [])[i] if "rain_sum" in daily_data else None,
                    "showers": daily_data.get("showers_sum", [])[i] if "showers_sum" in daily_data else None,
                    "snowfall": daily_data.get("snowfall_sum", [])[i] if "snowfall_sum" in daily_data else None,
                    "precipitation_hours": daily_data.get("precipitation_hours", [])[i] if "precipitation_hours" in daily_data else None,
                    "precipitation_probability": daily_data.get("precipitation_probability_max", [])[i] if "precipitation_probability_max" in daily_data else None,
                    "wind_speed": daily_data.get("wind_speed_10m_max", [])[i] if "wind_speed_10m_max" in daily_data else None,
                    "wind_gusts": daily_data.get("wind_gusts_10m_max", [])[i] if "wind_gusts_10m_max" in daily_data else None,
                    "wind_direction": daily_data.get("wind_direction_10m_dominant", [])[i] if "wind_direction_10m_dominant" in daily_data else None,
                    "radiation": daily_data.get("shortwave_radiation_sum", [])[i] if "shortwave_radiation_sum" in daily_data else None,
                    "weather_code": weather_code,
                    "weather_main": weather_main,
                    "is_rainy": weather_main in ["Rain", "Rain Showers", "Drizzle", "Freezing Rain", "Freezing Drizzle", "Thunderstorm"],
                    "is_snowy": weather_main in ["Snow", "Snow Showers"],
                    "is_sunny": weather_main in ["Clear", "Mainly Clear", "Partly Cloudy"],
                    "is_cloudy": weather_main in ["Partly Cloudy", "Overcast"]
                }
                
                # If 'apparent_temperature' is not available, use then compute feels_like from  min_feels_like and max_feels_like else from  apparent_temperature
                if "apparent_temperature" not in daily_data:
                    daily_forecast["feels_like"] = (daily_forecast["min_feels_like"] + daily_forecast["max_feels_like"]) / 2
                else:
                    daily_forecast["feels_like"] = daily_data.get("apparent_temperature", [])[i]
                
                forecasts.append(daily_forecast)
            
            # Cache and return the results
            self._set_cached_data(cache_key, forecasts)
            data_logger.info(f"Retrieved {len(forecasts)} days of forecast data")
            return forecasts
            
        except Exception as e:
            data_logger.error(f"Error getting forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []
    
    def get_historical_weather(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get historical weather data for Sheffield using Open-Meteo's archive API
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of dictionaries with daily historical weather data
        """
        try:
            # Check cache first
            cache_key = f"historical_{start_date}_{end_date}"
            is_valid, cached_data = self._get_cached_data(cache_key)
            
            if is_valid:
                data_logger.info(f"Using cached historical data for {start_date} to {end_date}")
                return cached_data
            
            # Check if end_date is today or in the future
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # If end_date is today or future, adjust it to yesterday to avoid issues
            if end_date_obj >= today:
                end_date_obj = today - timedelta(days=1)
                end_date = end_date_obj.strftime("%Y-%m-%d")
                data_logger.info(f"Adjusted end date to yesterday: {end_date}")
            
            # Make API request to Open-Meteo Archive
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "start_date": start_date,
                "end_date": end_date,
                "daily": [
                    "weather_code", 
                    "temperature_2m_max", 
                    "temperature_2m_min",
                    "apparent_temperature_max",
                    "apparent_temperature_min",
                    "precipitation_sum",
                    "rain_sum",
                    "snowfall_sum",
                    "precipitation_hours",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max",
                    "wind_direction_10m_dominant",
                    "shortwave_radiation_sum"
                ],
                "timezone": "GMT"  # Use GMT timezone for consistency
            }
            
            response = requests.get(self.historical_api_url, params=params)
            
            if response.status_code != 200:
                data_logger.error(f"Error from Open-Meteo Archive API: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            daily_data = data.get("daily", {})
            daily_time = daily_data.get("time", [])
            
            # Process the historical data
            historical_data = []
            
            for i in range(len(daily_time)):
                date_str = daily_time[i]
                weather_code = daily_data.get("weather_code", [])[i] if "weather_code" in daily_data else None
                weather_main, _ = self._get_weather_description(weather_code) if weather_code is not None else ("Unknown", "Unknown")
                
                daily_record = {
                    "date": date_str,
                    "min_temp": daily_data.get("temperature_2m_min", [])[i] if "temperature_2m_min" in daily_data else None,
                    "max_temp": daily_data.get("temperature_2m_max", [])[i] if "temperature_2m_max" in daily_data else None,
                    "avg_temp": (daily_data.get("temperature_2m_min", [])[i] + daily_data.get("temperature_2m_max", [])[i]) / 2 if "temperature_2m_min" in daily_data and "temperature_2m_max" in daily_data else None,
                    "min_feels_like": daily_data.get("apparent_temperature_min", [])[i] if "apparent_temperature_min" in daily_data else None,
                    "max_feels_like": daily_data.get("apparent_temperature_max", [])[i] if "apparent_temperature_max" in daily_data else None,
                    "precipitation": daily_data.get("precipitation_sum", [])[i] if "precipitation_sum" in daily_data else None,
                    "rain": daily_data.get("rain_sum", [])[i] if "rain_sum" in daily_data else None,
                    "snowfall": daily_data.get("snowfall_sum", [])[i] if "snowfall_sum" in daily_data else None,
                    "precipitation_hours": daily_data.get("precipitation_hours", [])[i] if "precipitation_hours" in daily_data else None,
                    "wind_speed": daily_data.get("wind_speed_10m_max", [])[i] if "wind_speed_10m_max" in daily_data else None,
                    "wind_gusts": daily_data.get("wind_gusts_10m_max", [])[i] if "wind_gusts_10m_max" in daily_data else None,
                    "wind_direction": daily_data.get("wind_direction_10m_dominant", [])[i] if "wind_direction_10m_dominant" in daily_data else None,
                    "radiation": daily_data.get("shortwave_radiation_sum", [])[i] if "shortwave_radiation_sum" in daily_data else None,
                    "weather_code": weather_code,
                    "weather_main": weather_main,
                    "is_rainy": weather_main in ["Rain", "Rain Showers", "Drizzle", "Freezing Rain", "Freezing Drizzle", "Thunderstorm"],
                    "is_snowy": weather_main in ["Snow", "Snow Showers"],
                    "is_sunny": weather_main in ["Clear", "Mainly Clear", "Partly Cloudy"],
                    "is_cloudy": weather_main in ["Partly Cloudy", "Overcast"],
                    "humidity": 65  # Estimate as humidity not directly available in daily data
                }

                # If 'apparent_temperature' is not available, use then compute feels_like from  min_feels_like and max_feels_like else from  apparent_temperature
                if "apparent_temperature" not in daily_data:
                    daily_record["feels_like"] = (daily_record["min_feels_like"] + daily_record["max_feels_like"]) / 2
                else:
                    daily_record["feels_like"] = daily_data.get("apparent_temperature", [])[i]
                
                historical_data.append(daily_record)
            
            # Cache the result
            self._set_cached_data(cache_key, historical_data)
            data_logger.info(f"Retrieved {len(historical_data)} days of historical weather data")
            
            return historical_data
            
        except Exception as e:
            data_logger.error(f"Error getting historical weather: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []
    '''
    def prepare_weather_for_prophet(self, start_date: str, end_date: str, forecast_days: int = 30) -> pd.DataFrame:
        """
        Prepare weather data for Prophet model (historical + forecast)
        
        Args:
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with enhanced weather data ready for Prophet
        """
        try:
            # Get historical data
            historical = self.get_historical_weather(start_date, end_date)
            
            if not historical:
                data_logger.warning(f"No historical weather data available for {start_date} to {end_date}.")
                # Return empty DataFrame with proper structure
                empty_columns = [
                    'ds', 'temperature', 'temperature_min', 'temperature_max', 
                    'precipitation', 'wind_speed', 'humidity', 'radiation',
                    'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy', 
                    'temp_delta', 'feels_like', 'weather_code',
                    'precipitation_hours', 'precipitation_intensity'
                ]
                return pd.DataFrame(columns=empty_columns)
            

            
            # Get forecast data
            forecast = self.get_forecast(forecast_days)
            
            # Combine data - handle case where forecast might be None
            if forecast:
                combined = historical + forecast
            else:
                data_logger.warning(f"No forecast weather data available. Using only historical data.")
                combined = historical
            

            # Convert to DataFrame
            df = pd.DataFrame(combined)
            
            # Ensure date is properly formatted
            df['ds'] = pd.to_datetime(df['date'])
            
            # Create derived features
            
            # 1. Temperature delta (daily temperature range)
            df['temp_delta'] = df['max_temp'] - df['min_temp']
            
            # 2. Average feels like temperature
            df['feels_like'] = (df['min_feels_like'] + df['max_feels_like']) / 2
            
            # 3. Calculate precipitation intensity (mm per hour when it rains)
            df['precipitation_intensity'] = df.apply(
                lambda x: x['precipitation'] / max(x['precipitation_hours'], 1) if pd.notnull(x['precipitation']) and pd.notnull(x['precipitation_hours']) else 0, 
                axis=1
            )
            
            # 4. Weather pattern types - convert to numeric (already added in get_historical/forecast)
            df['is_rainy'] = df['is_rainy'].astype(int)
            df['is_snowy'] = df['is_snowy'].astype(int)
            df['is_sunny'] = df['is_sunny'].astype(int)
            df['is_cloudy'] = df['is_cloudy'].astype(int)
            
            # 5. Select and rename final columns for Prophet
            prophet_df = df[[
                'ds', 'avg_temp', 'min_temp', 'max_temp', 
                'precipitation', 'wind_speed', 'humidity', 'radiation',
                'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy', 
                'temp_delta', 'feels_like', 'weather_code',
                'precipitation_hours', 'precipitation_intensity'
            ]].copy()
            
            # Rename to standardized names
            prophet_df = prophet_df.rename(columns={
                'avg_temp': 'temperature'
            })
            
            # Fill any remaining NaN values with appropriate defaults
            # Use appropriate imputation strategies for each column
            for col in prophet_df.columns:
                if col == 'ds':
                    continue
                    
                # Different imputation strategies based on the variable type
                if col in ['temperature', 'min_temp', 'max_temp', 'feels_like']:
                    # For temperature variables, use the mean
                    prophet_df[col] = prophet_df[col].fillna(prophet_df[col].mean())
                elif col in ['precipitation', 'precipitation_hours', 'precipitation_intensity']:
                    # For precipitation variables, use 0
                    prophet_df[col] = prophet_df[col].fillna(0)
                elif col in ['is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy']:
                    # For binary flags, use 0
                    prophet_df[col] = prophet_df[col].fillna(0)
                else:
                    # For other variables, use the median (more robust to outliers)
                    prophet_df[col] = prophet_df[col].fillna(prophet_df[col].median())
            
            data_logger.info(f"Prepared enhanced weather data for Prophet with {len(prophet_df)} rows")
            return prophet_df
            
        except Exception as e:
            data_logger.error(f"Error preparing weather for Prophet: {str(e)}")
            data_logger.error(traceback.format_exc())
            # Return empty DataFrame with proper structure
            empty_columns = [
                'ds', 'temperature', 'temperature_min', 'temperature_max', 
                'precipitation', 'wind_speed', 'humidity', 'radiation',
                'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy', 
                'temp_delta', 'feels_like', 'weather_code',
                'precipitation_hours', 'precipitation_intensity'
            ]
            return pd.DataFrame(columns=empty_columns)
    '''

    def prepare_weather_for_prophet(self, start_date: str, end_date: str, forecast_days: int = 30) -> pd.DataFrame:
        """
        Prepare weather data for Prophet model (historical + forecast)
        
        Args:
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with enhanced weather data ready for Prophet
        """
        try:
            # Get historical data
            historical = self.get_historical_weather(start_date, end_date)
            
            if not historical:
                data_logger.warning(f"No historical weather data available for {start_date} to {end_date}.")
                # Return empty DataFrame with proper structure
                empty_columns = [
                'ds', 'temperature', 'min_temp', 'max_temp', 'min_feels_like', 'max_feels_like', 'rain', 
                'snowfall',  'precipitation', 'wind_speed', 'radiation',
                'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy', 
                'temp_delta', 'feels_like', 
                'precipitation_hours', 'precipitation_intensity'
                ]
                return pd.DataFrame(columns=empty_columns)
            
            # Get forecast data
            forecast = self.get_forecast(forecast_days)
            
            # Calculate the gap between historical end_date and forecast start date
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # If there's a gap between end_date and today (start of forecast), fill it with historical data
            if end_date_obj < today:
                # Calculate the gap period
                gap_start = (end_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
                gap_end = (today - timedelta(days=1)).strftime("%Y-%m-%d")
                
                # Only fetch gap data if there's actually a gap (at least one day)
                if datetime.strptime(gap_start, "%Y-%m-%d") <= datetime.strptime(gap_end, "%Y-%m-%d"):
                    data_logger.info(f"Fetching historical data for gap period: {gap_start} to {gap_end}")
                    try:
                        gap_data = self.get_historical_weather(gap_start, gap_end)
                    except:
                        try:
                            gap_end = (today - timedelta(days=2)).strftime("%Y-%m-%d")
                            gap_data = self.get_historical_weather(gap_start, gap_end)
                        except:
                            gap_end = (today - timedelta(days=3)).strftime("%Y-%m-%d")
                            gap_data = self.get_historical_weather(gap_start, gap_end)                            

                    # Add the gap data to historical
                    if gap_data:
                        historical.extend(gap_data)
                        data_logger.info(f"Added {len(gap_data)} days of historical data to fill the gap")
            
            # Combine data - handle case where forecast might be None
            if forecast:
                combined = historical + forecast

            else:
                data_logger.warning(f"No forecast weather data available. Using only historical data.")
                combined = historical
            
            # Convert to DataFrame
            df = pd.DataFrame(combined)

            # save the dataframe to a csv file
            #df.to_csv(f"weather_data.csv", index=False)
            
            # Ensure date is properly formatted
            df['ds'] = pd.to_datetime(df['date'])
            
            # Create derived features
            
            # 1. Temperature delta (daily temperature range)
            df['temp_delta'] = df['max_temp'] - df['min_temp']
            
            # 2. Average feels like temperature
            
            if 'feels_like' in df.columns:
                df['feels_like'] = df['feels_like']
            else:
                df['feels_like'] = (df['min_feels_like'] + df['max_feels_like']) / 2

            # 3. Calculate precipitation intensity (mm per hour when it rains)
            df['precipitation_intensity'] = df.apply(
                lambda x: x['precipitation'] / max(x['precipitation_hours'], 1) if pd.notnull(x['precipitation']) and pd.notnull(x['precipitation_hours']) else 0, 
                axis=1
            )
            
            # 4. Weather pattern types - convert to numeric (already added in get_historical/forecast)
            df['is_rainy'] = df['is_rainy'].astype(int)
            df['is_snowy'] = df['is_snowy'].astype(int)
            df['is_sunny'] = df['is_sunny'].astype(int)
            df['is_cloudy'] = df['is_cloudy'].astype(int)
            
            # 5. Select and rename final columns for Prophet
            prophet_df = df[[
                'ds', 'avg_temp', 'min_temp', 'max_temp', 'min_feels_like', 'max_feels_like', 'rain', 
                'snowfall',  'precipitation', 'wind_speed', 'radiation',
                'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy', 
                'temp_delta', 'feels_like', 
                'precipitation_hours', 'precipitation_intensity'
            ]].copy()
            
            # Rename to standardized names
            prophet_df = prophet_df.rename(columns={
                'avg_temp': 'temperature'
            })
            
            # Fill any remaining NaN values with appropriate defaults
            # Use appropriate imputation strategies for each column
            for col in prophet_df.columns:
                if col == 'ds':
                    continue
                    
                # Different imputation strategies based on the variable type
                if col in ['temperature', 'min_temp', 'max_temp', 'min_feels_like', 'max_feels_like','feels_like', 'temp_delta']:
                    # For these variables, use the mean
                    prophet_df[col] = prophet_df[col].fillna(prophet_df[col].mean())
                elif col in ['precipitation', 'precipitation_hours', 'precipitation_intensity', 'rain', 'snowfall', 'radiation']:
                    # For precipitation variables, use 0
                    prophet_df[col] = prophet_df[col].fillna(0)
                elif col in ['is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy']:
                    # For binary flags, use 0
                    prophet_df[col] = prophet_df[col].fillna(0)
                else:
                    # For other variables, use the median (more robust to outliers)
                    prophet_df[col] = prophet_df[col].fillna(prophet_df[col].median())
            
            data_logger.info(f"Prepared enhanced weather data for Prophet with {len(prophet_df)} rows")

            # save the prophet_df to a csv file
            #prophet_df.to_csv(f"weather_data_prophet.csv", index=False)

            return prophet_df
            
        except Exception as e:
            data_logger.error(f"Error preparing weather for Prophet: {str(e)}")
            data_logger.error(traceback.format_exc())
            # Return empty DataFrame with proper structure
            empty_columns = [
                'ds', 'temperature', 'temperature_min', 'temperature_max', 
                'precipitation', 'wind_speed', 'humidity', 'radiation',
                'is_rainy', 'is_snowy', 'is_sunny', 'is_cloudy', 
                'temp_delta', 'feels_like', 'weather_code',
                'precipitation_hours', 'precipitation_intensity'
            ]
            return pd.DataFrame(columns=empty_columns)
        
        
def test_weather_service():
    weather_service = WeatherService()
    print(weather_service.get_current_weather())
    print(weather_service.get_forecast())
    print(weather_service.get_historical_weather("2024-01-01", "2024-01-31"))
    print(weather_service.prepare_weather_for_prophet("2024-01-01", "2024-01-31"))

if __name__ == "__main__":
    test_weather_service()



