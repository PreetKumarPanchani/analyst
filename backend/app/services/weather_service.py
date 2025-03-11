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
    Uses the OpenWeatherMap API for current and forecast data
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = None):
        """Initialize the weather service"""
        self.api_key = api_key or settings.WEATHER_API_KEY
        self.api_base_url = settings.WEATHER_API_URL
        
        # Sheffield coordinates
        self.lat = settings.LOCATION_COORDINATES["sheffield"]["lat"]
        self.lon = settings.LOCATION_COORDINATES["sheffield"]["lon"]
        
        # Cache settings
        self.cache_dir = cache_dir or os.path.join(settings.PROCESSED_DATA_DIR, "cache")
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
        Get current weather data for Sheffield
        
        Returns:
            Dictionary with weather data
        """
        try:
            # Check cache first
            cache_key = "current_weather"
            is_valid, cached_data = self._get_cached_data(cache_key)
            
            if is_valid:
                return cached_data
            
            # Exit early if no API key
            if not self.api_key:
                data_logger.warning("No OpenWeatherMap API key provided")
                return None
            # Make API request
            url = f"{self.api_base_url}/weather"
            params = {
                "lat": self.lat,
                "lon": self.lon,
                "appid": self.api_key,
                "units": "metric"  # Use metric units
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                data_logger.error(f"Error from OpenWeatherMap API: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            
            # Simplify the response
            weather_data = {
                "timestamp": datetime.now().isoformat(),
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "weather_main": data["weather"][0]["main"],
                "weather_description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "clouds": data["clouds"]["all"],
                "rain_1h": data.get("rain", {}).get("1h", 0),
                "snow_1h": data.get("snow", {}).get("1h", 0)
            }
            
            # Cache the result
            self._set_cached_data(cache_key, weather_data)
            
            data_logger.info(f"Retrieved current weather: {weather_data['temperature']}°C, {weather_data['weather_main']}")
            
            return weather_data
            
        except Exception as e:
            data_logger.error(f"Error getting current weather: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None
        
    def get_forecast(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get weather forecast for Sheffield
        Tries to use daily forecast API first, falls back to 3-hourly if unavailable
        
        Args:
            days: Number of days to forecast (max 16 for paid API, max 5-7 for free API)
            
        Returns:
            List of dictionaries with daily forecast data
        """
        try:
            # Check cache first
            cache_key = f"forecast_{days}"
            is_valid, cached_data = self._get_cached_data(cache_key)
            
            if is_valid:
                return cached_data
            
            # Exit early if no API key
            if not self.api_key:
                data_logger.warning("No OpenWeatherMap API key provided")
                return None
            
            # First try the daily forecast endpoint (requires paid subscription)
            url = f"{self.api_base_url}/forecast/daily"
            params = {
                "lat": self.lat,
                "lon": self.lon,
                "cnt": min(days, 16),
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params)
            
            # If successful, process the daily forecast data
            if response.status_code == 200:
                data = response.json()
                
                # Process daily forecast data
                forecasts = []
                
                for item in data["list"]:
                    dt = datetime.fromtimestamp(item["dt"])
                    date_str = dt.strftime("%Y-%m-%d")
                    
                    daily_forecast = {
                        "date": date_str,
                        "min_temp": item["temp"]["min"],
                        "max_temp": item["temp"]["max"],
                        "avg_temp": (item["temp"]["day"] + item["temp"]["night"]) / 2,
                        "humidity": item["humidity"],
                        "pressure": item["pressure"],
                        "wind_speed": item["speed"],
                        "precipitation": item.get("rain", 0) + item.get("snow", 0),
                        "weather_main": item["weather"][0]["main"],
                        "is_rainy": item["weather"][0]["main"] in ["Rain", "Drizzle", "Thunderstorm"],
                        "is_sunny": item["weather"][0]["main"] in ["Clear", "Sunny"],
                        "is_snowy": item["weather"][0]["main"] == "Snow",
                        "pop": item.get("pop", 0),  # Probability of precipitation
                    }
                    
                    forecasts.append(daily_forecast)
                    
                # Cache and return the results
                self._set_cached_data(cache_key, forecasts)
                data_logger.info(f"Retrieved {len(forecasts)} days of forecast data from daily API")
                return forecasts
            
            # If daily forecast API not available, fall back to 3-hourly API
            data_logger.info("Daily forecast API not available, falling back to 3-hourly forecast")
            
            # Make API request to 3-hourly forecast endpoint (free tier)
            url = f"{self.api_base_url}/forecast"
            params = {
                "lat": self.lat,
                "lon": self.lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                data_logger.error(f"Error from OpenWeatherMap API: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            
            # Process 3-hourly forecasts to daily (with improvements)
            forecasts = []
            forecast_by_day = {}
            
            for item in data["list"]:
                # Convert timestamp to date
                dt = datetime.fromtimestamp(item["dt"])
                date_str = dt.strftime("%Y-%m-%d")
                
                # Group by day
                if date_str not in forecast_by_day:
                    forecast_by_day[date_str] = []
                
                forecast_by_day[date_str].append({
                    "datetime": dt.isoformat(),
                    "temperature": item["main"]["temp"],
                    "feels_like": item["main"]["feels_like"],
                    "humidity": item["main"]["humidity"],
                    "pressure": item["main"]["pressure"],
                    "weather_main": item["weather"][0]["main"],
                    "weather_description": item["weather"][0]["description"],
                    "wind_speed": item["wind"]["speed"],
                    "wind_gust": item["wind"].get("gust", 0),
                    "clouds": item["clouds"]["all"],
                    "rain_3h": item.get("rain", {}).get("3h", 0),
                    "snow_3h": item.get("snow", {}).get("3h", 0),
                    "pop": item.get("pop", 0),  # Probability of precipitation
                    "hour": dt.hour
                })
            
            # Aggregate to daily forecasts (improved aggregation)
            for date_str, items in forecast_by_day.items():
                temps = [item["temperature"] for item in items]
                
                # More efficient way to find most common weather
                weather_counts = {}
                for item in items:
                    w_main = item["weather_main"]
                    weather_counts[w_main] = weather_counts.get(w_main, 0) + 1
                
                daily_forecast = {
                    "date": date_str,
                    "min_temp": min(temps),
                    "max_temp": max(temps),
                    "avg_temp": sum(temps) / len(temps),
                    "morning_temp": next((item["temperature"] for item in items if 7 <= item["hour"] <= 10), None),
                    "evening_temp": next((item["temperature"] for item in items if 17 <= item["hour"] <= 20), None),
                    "humidity": sum(item["humidity"] for item in items) / len(items),
                    "pressure": sum(item["pressure"] for item in items) / len(items),
                    "wind_speed": max(item["wind_speed"] for item in items),  # Use max wind speed
                    "wind_gust": max((item.get("wind_gust", 0) for item in items), default=0),
                    "precipitation": sum(item.get("rain_3h", 0) + item.get("snow_3h", 0) for item in items),
                    "pop": max(item.get("pop", 0) for item in items),  # Max probability of precipitation
                    "weather_main": max(weather_counts.items(), key=lambda x: x[1])[0],
                    "is_rainy": any(item["weather_main"] in ["Rain", "Drizzle", "Thunderstorm"] for item in items),
                    "is_sunny": any(item["weather_main"] in ["Clear", "Sunny"] and 7 <= item["hour"] <= 19 for item in items),
                    "is_snowy": any(item["weather_main"] == "Snow" for item in items),
                }
                
                forecasts.append(daily_forecast)
            
            # Sort by date and limit to requested days
            forecasts.sort(key=lambda x: x["date"])
            forecasts = forecasts[:days]
            
            # Cache the result
            self._set_cached_data(cache_key, forecasts)
            
            data_logger.info(f"Retrieved {len(forecasts)} days of forecast data from 3-hourly API")
            
            return forecasts
            
        except Exception as e:
            data_logger.error(f"Error getting forecast: {str(e)}")
            data_logger.error(traceback.format_exc())
            return None
        
    '''
    def get_historical_weather(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get historical weather data for Sheffield from Open-Meteo API, aggregated to daily summaries
        
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

            # Construct Open-Meteo API URL for hourly data
            url = (f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={self.lat}&longitude={self.lon}&"
                f"start_date={start_date}&end_date={end_date}&"
                f"hourly=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m")

            # Make API request
            response = requests.get(url)
            if response.status_code != 200:
                data_logger.error(f"Error from Open-Meteo API: {response.status_code} - {response.text}")
                return []

            data = response.json()
            hourly_data = data.get("hourly", {})
            
            # Extract hourly data
            times = hourly_data.get("time", [])
            temps = hourly_data.get("temperature_2m", [])
            precipitations = hourly_data.get("precipitation", [])
            humidities = hourly_data.get("relative_humidity_2m", [])
            wind_speeds = hourly_data.get("wind_speed_10m", [])

            # Aggregate hourly data into daily summaries
            daily_data = {}
            for i, timestamp in enumerate(times):
                date = timestamp.split("T")[0]  # Extract date (YYYY-MM-DD)
                if date not in daily_data:
                    daily_data[date] = {
                        "temps": [],
                        "precipitations": [],
                        "humidities": [],
                        "wind_speeds": []
                    }
                daily_data[date]["temps"].append(temps[i])
                daily_data[date]["precipitations"].append(precipitations[i])
                daily_data[date]["humidities"].append(humidities[i])
                daily_data[date]["wind_speeds"].append(wind_speeds[i])

            # Build daily historical data list
            historical_data = []
            for date, values in daily_data.items():
                                # Filter out None values before summing
                precipitations = [p for p in values["precipitations"] if p is not None]
                total_precip = sum(precipitations) if precipitations else 0
                
                weather_main = "Rain" if total_precip > 0 else "Clear"
                daily_dict = {
                    "date": date,
                    "min_temp": min(values["temps"]),
                    "max_temp": max(values["temps"]),
                    "avg_temp": sum(values["temps"]) / len(values["temps"]),
                    "humidity": sum(values["humidities"]) / len(values["humidities"]),
                    "pressure": 1013,  # Default value since Open-Meteo doesn't provide pressure
                    "wind_speed": max(values["wind_speeds"]),  # Use max wind speed
                    "precipitation": total_precip,
                    "weather_main": weather_main,
                    "is_rainy": total_precip > 0,
                    "is_sunny": total_precip == 0,
                    "is_snowy": False  # Simplified; could check temp < 0 and precip > 0
                }
                historical_data.append(daily_dict)

            # Cache the result
            self._set_cached_data(cache_key, historical_data)
            data_logger.info(f"Retrieved {len(historical_data)} days of historical weather data from Open-Meteo")

            return historical_data

        except Exception as e:
            data_logger.error(f"Error getting historical weather from Open-Meteo: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []
    '''

        

    def get_historical_weather(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get historical weather data for Sheffield from Open-Meteo API, aggregated to daily summaries
        
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

            # Construct Open-Meteo API URL for hourly data
            url = (f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={self.lat}&longitude={self.lon}&"
                f"start_date={start_date}&end_date={end_date}&"
                f"hourly=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m")

            # Make API request
            response = requests.get(url)
            if response.status_code != 200:
                data_logger.error(f"Error from Open-Meteo API: {response.status_code} - {response.text}")
                return []

            data = response.json()
            hourly_data = data.get("hourly", {})
            
            # Extract hourly data
            times = hourly_data.get("time", [])
            temps = hourly_data.get("temperature_2m", [])
            precipitations = hourly_data.get("precipitation", [])
            humidities = hourly_data.get("relative_humidity_2m", [])
            wind_speeds = hourly_data.get("wind_speed_10m", [])

            # Aggregate hourly data into daily summaries
            daily_data = {}
            for i, timestamp in enumerate(times):
                if i >= len(temps) or i >= len(precipitations) or i >= len(humidities) or i >= len(wind_speeds):
                    continue  # Skip if any data is missing for this timestamp
                    
                date = timestamp.split("T")[0]  # Extract date (YYYY-MM-DD)
                if date not in daily_data:
                    daily_data[date] = {
                        "temps": [],
                        "precipitations": [],
                        "humidities": [],
                        "wind_speeds": []
                    }
                    
                # Only add non-None values
                if temps[i] is not None:
                    daily_data[date]["temps"].append(temps[i])
                if precipitations[i] is not None:
                    daily_data[date]["precipitations"].append(precipitations[i])
                if humidities[i] is not None:
                    daily_data[date]["humidities"].append(humidities[i])
                if wind_speeds[i] is not None:
                    daily_data[date]["wind_speeds"].append(wind_speeds[i])

            # Build daily historical data list
            historical_data = []
            for date, values in daily_data.items():
                # Skip days with no temperature data
                if not values["temps"]:
                    continue
                    
                total_precip = sum(values["precipitations"]) if values["precipitations"] else 0
                weather_main = "Rain" if total_precip > 0 else "Clear"
                
                # Safe calculations with null checking
                daily_dict = {
                    "date": date,
                    "min_temp": min(values["temps"]) if values["temps"] else 0,
                    "max_temp": max(values["temps"]) if values["temps"] else 0,
                    "avg_temp": sum(values["temps"]) / len(values["temps"]) if values["temps"] else 0,
                    "humidity": sum(values["humidities"]) / len(values["humidities"]) if values["humidities"] else 50,
                    "pressure": 1013,  # Default value since Open-Meteo doesn't provide pressure
                    "wind_speed": max(values["wind_speeds"]) if values["wind_speeds"] else 0,
                    "precipitation": total_precip,
                    "weather_main": weather_main,
                    "is_rainy": total_precip > 0,
                    "is_sunny": total_precip == 0,
                    "is_snowy": False  # Simplified; could check temp < 0 and precip > 0
                }
                historical_data.append(daily_dict)

            # Cache the result
            self._set_cached_data(cache_key, historical_data)
            data_logger.info(f"Retrieved {len(historical_data)} days of historical weather data from Open-Meteo")

            return historical_data

        except Exception as e:
            data_logger.error(f"Error getting historical weather from Open-Meteo: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []

    
    def prepare_weather_for_prophet(self, start_date: str, end_date: str, forecast_days: int = 30) -> pd.DataFrame:
        """
        Prepare weather data for Prophet model (historical + forecast)
        
        Args:
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with weather data ready for Prophet
        """
        try:
            # Get historical data
            historical = self.get_historical_weather(start_date, end_date)
            
            if not historical:
                data_logger.warning(f"No historical weather data available for {start_date} to {end_date}.")
                # Don't use synthetic data, return empty DataFrame with proper structure
                return pd.DataFrame(columns=['ds', 'temperature', 'precipitation', 'rainy', 'sunny', 'temperature_squared'])
            
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
            
            # Select and rename columns for Prophet
            prophet_df = df[['ds', 'avg_temp', 'precipitation', 'is_rainy', 'is_sunny']].copy()
            
            # Fill NaN values before converting to int
            for col in ['is_rainy', 'is_sunny']:
                # First fill NaN values with 0
                prophet_df[col] = prophet_df[col].fillna(0)
                # Then convert to int
                prophet_df[col] = prophet_df[col].astype(int)
            
            # Rename to standardized names
            prophet_df = prophet_df.rename(columns={
                'avg_temp': 'temperature',
                'is_rainy': 'rainy',
                'is_sunny': 'sunny'
            })
            
            # Add temperature squared for non-linear effects
            prophet_df['temperature_squared'] = prophet_df['temperature'] ** 2
            
            data_logger.info(f"Prepared weather data for Prophet with {len(prophet_df)} rows")
            
            return prophet_df
            
        except Exception as e:
            data_logger.error(f"Error preparing weather for Prophet: {str(e)}")
            data_logger.error(traceback.format_exc())
            # Return empty DataFrame with proper structure 
            return pd.DataFrame(columns=['ds', 'temperature', 'precipitation', 'rainy', 'sunny', 'temperature_squared'])

# Test function
def test_weather_service():
    """Test the WeatherService functionality"""
    service = WeatherService()
    
    # Test current weather
    current = service.get_current_weather()
    print(f"Current weather: {current['temperature']}°C, {current['weather_main']}")
    
    # Test forecast
    forecast = service.get_forecast(days=5)
    print(f"5-day forecast: {len(forecast)} days")
    for day in forecast:
        print(f"{day['date']}: {day['min_temp']}°C to {day['max_temp']}°C, {day['weather_main']}")
    
    # Test historical
    historical = service.get_historical_weather('2024-01-01', '2025-02-28')
    print(f"Historical data: {len(historical)} days")
    
    # Test preparation for Prophet
    prophet_data = service.prepare_weather_for_prophet('2024-01-01', '2025-02-28', 30)
    print(f"Prophet data: {len(prophet_data)} rows")
    print(f"Columns: {prophet_data.columns.tolist()}")
    print(f"Sample data:\n{prophet_data.head()}")
    
    return "Weather service test completed successfully"

if __name__ == "__main__":
    test_weather_service()

