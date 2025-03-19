import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import traceback
from typing import Dict, List, Optional, Any, Tuple
import sys
import holidays  # Make sure to import holidays
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.logger import data_logger
from app.core.config import settings
from app.services.s3_service import S3Service

class EventsService:
    """
    Service for retrieving and managing Sheffield event data
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize the events service"""
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "events.json")
        self.s3_service = S3Service()
        self.use_s3 = settings.USE_S3_STORAGE
        
        if not self.use_s3:
            os.makedirs(cache_dir, exist_ok=True)
            
        data_logger.info(f"EventsService initialized with cache directory: {cache_dir}")
    
    def _load_events_data(self) -> List[Dict[str, Any]]:
        """
        Load events data from the cache file or create default data
        
        Returns:
            List of dictionaries with event data
        """
        try:
            if self.use_s3:
                # Load from S3
                s3_key = f"{settings.S3_CACHE_PREFIX}events.json"
                
                if not self.s3_service.file_exists(s3_key):
                    data_logger.info("No events cache found in S3, getting default events")
                    # No cached events, get defaults
                    return self._create_default_events()
                
                # Get the JSON from S3
                response = self.s3_service.s3_client.get_object(
                    Bucket=self.s3_service.bucket_name,
                    Key=s3_key
                )
                json_str = response['Body'].read().decode('utf-8')
                
                # Load JSON
                events = json.loads(json_str)
                data_logger.info(f"Loaded {len(events)} events from S3 cache")
                return events
            else:
                # Load from local filesystem
                if not os.path.exists(self.cache_file):
                    data_logger.info("No events cache found, getting default events")
                    # No cached events, get defaults
                    return self._create_default_events()
                
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Validate that we have a list of dictionaries
                    if isinstance(data, list):
                        # Further validate structure of each item
                        valid_data = True
                        for item in data:
                            if not isinstance(item, dict) or 'name' not in item or 'date' not in item:
                                valid_data = False
                                break
                        if valid_data:
                            return data
                    # If we got here, the data is invalid
                    data_logger.warning("Cache file exists but contains invalid data. Creating default events.")
            
            # If no cache file exists or invalid data, create default Sheffield events
            return self._create_default_events()
            
        except Exception as e:
            data_logger.error(f"Error loading events data: {str(e)}")
            data_logger.error(traceback.format_exc())
            return self._create_default_events()
    
    def _save_events_data(self, events: List[Dict[str, Any]]) -> bool:
        """
        Save events data to the cache file
        
        Args:
            events: List of dictionaries with event data
            
        Returns:
            Success flag
        """
        try:
            if self.use_s3:
                # Save to S3
                s3_key = f"{settings.S3_CACHE_PREFIX}events.json"
                
                # Convert to JSON string
                json_str = json.dumps(events, indent=2)
                
                # Use put_object directly
                self.s3_service.s3_client.put_object(
                    Bucket=self.s3_service.bucket_name,
                    Key=s3_key,
                    Body=json_str
                )
                
                data_logger.info(f"Saved {len(events)} events to S3 cache")
            else:
                # Save to local filesystem
                with open(self.cache_file, 'w') as f:
                    json.dump(events, f, indent=2)
                
                data_logger.info(f"Saved {len(events)} events to cache")
            
            return True
            
        except Exception as e:
            data_logger.error(f"Error saving events data: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def _get_uk_holidays(self, start_year: int = 2024, end_year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get UK bank holidays using the holidays library
        
        Args:
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive, None means all available future years)
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries with holiday data
        """
        try:
            if end_year is None:
                end_year = start_year 
            
            # Generate years list
            years = list(range(start_year, end_year + 1))
            
            data_logger.info(f"Retrieving UK holidays for years: {years}")
            
            # Initialize UK holidays with the specified years
            uk_holidays = holidays.country_holidays('GB', subdiv='ENG', years=years)
            
            # Collect all holidays for the specified years
            holiday_list = []
            for date, name in sorted(uk_holidays.items()):
                holiday_list.append({
                    "name": name,
                    "date": date.strftime("%Y-%m-%d"),
                    "type": "Holiday",
                    "description": "UK Bank Holiday"
                })
            
            print(f'Retrieved {len(holiday_list)} UK holidays for years: {years}')
            print('holiday_list::::::::',holiday_list)

            data_logger.info(f"Retrieved {len(holiday_list)} UK holidays for years: {years}")
            return holiday_list
            
        except Exception as e:
            data_logger.error(f"Error getting UK holidays from library: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []
    
    def _get_major_events(self, start_year: int = 2024, end_year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get manually curated list of major events in England and Sheffield
        
        Args:
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive, None means all available future years)
            
        Returns:
            List[Dict[str, Any]]: List of major events
        """
        # Major events that are confirmed for 2024-2025
        major_events = [
 
            {
                "name": "Tramlines Festival 2024",
                "date": "2024-07-26",
                "end_date": "2024-07-28",
                "type": "Festival",
                "description": "Sheffield's biggest city music festival"
            },
          
            # 2024 Sheffield-specific Events
            {
                "name": "Sheffield Beer Week",
                "date": "2024-03-04",
                "end_date": "2024-03-10",
                "type": "Festival",
                "description": "Week-long beer festival across Sheffield"
            },
            {
                "name": "The Crucible Tournament Initial Days",
                "date": "2024-04-20",
                "end_date": "2024-04-23",
                "type": "Festival",
                "description": "Snooker tournament in Sheffield"
            },
            {
                "name": "The Crucible Tournament Final Days",
                "date": "2024-05-04",
                "end_date": "2024-05-06",
                "type": "Festival",
                "description": "Snooker tournament in Sheffield"
            },
            {
                "name": "Sheffield Half Marathon",
                "date": "2024-04-07",
                "type": "Festival",
                "description": "Half marathon through Sheffield"
            },
            {
                "name": "Sheffield DocFest",
                "date": "2024-06-12",
                "end_date": "2024-06-17",
                "type": "Festival",
                "description": "Documentary festival in Sheffield"
            },
            {
                "name": "Sheffield Chamber Music Festival",
                "date": "2024-05-17",
                "end_date": "2024-05-25",
                "type": "Festival",
                "description": "Chamber music festival in Sheffield"
            },
            {
                "name": "Sheffield Food Festival",
                "date": "2024-05-26",
                "end_date": "2024-05-28",
                "type": "Festival",
                "description": "Food festival in Sheffield city center"
            },

            {
                "name": "Sheffield Christmas Initial Days Market",
                "date": "2024-11-14",
                "end_date": "2024-11-16",
                "type": "Festival",
                "description": "Christmas market in Sheffield city center"
            },
            {
                "name": "Sheffield Christmas Final Days Market",
                "date": "2024-11-22",
                "end_date": "2024-11-24",
                "type": "Festival",
                "description": "Christmas market in Sheffield city center"
            },
            {
                "name": "Sheffield Radical Pride",
                "date": "2024-06-23",
                "type": "Festival",
                "description": "LGBTIQ+ celebration in Sheffield"
            },
            
            # 2024 Observance Days
            {
                "name": "Mother's Day 2024",
                "date": "2024-03-10",
                "type": "Festival",
                "description": "Mothering Sunday in the UK (4th Sunday of Lent)"
            },
            {
                "name": "Father's Day 2024",
                "date": "2024-06-16",
                "type": "Festival",
                "description": "Day to honor fathers (3rd Sunday in June)"
            },
            {
                "name": "Valentine's Day 2024",
                "date": "2024-02-14",
                "type": "Festival",
                "description": "Day celebrating love and romance"
            },
            {
                "name": "St. Patrick's Day 2024",
                "date": "2024-03-17",
                "type": "Festival",
                "description": "Celebration of Irish culture and heritage"
            },
            {
                "name": "St. George's Day 2024",
                "date": "2024-04-23",
                "type": "Festival",
                "description": "England's national day"
            },
            {
                "name": "Remembrance Day 2024",
                "date": "2024-11-11",
                "type": "Festival",
                "description": "Armistice Day commemorating the end of World War I"
            },
            {
                "name": "Remembrance Sunday 2024",
                "date": "2024-11-10",
                "type": "Festival",
                "description": "Sunday closest to November 11th honoring those who died in the line of duty"
            },
            {
                "name": "Halloween 2024",
                "date": "2024-10-31",
                "type": "Festival",
                "description": "Evening before All Hallows' Day"
            },
            {
                "name": "Pancake Day 2024",
                "date": "2024-02-13",
                "type": "Festival",
                "description": "Shrove Tuesday, the day before Ash Wednesday"
            },
            {
                "name": "Chinese New Year 2024",
                "date": "2024-02-10",
                "type": "Festival",
                "description": "Year of the Dragon celebrations"
            },
            {
                "name": "Diwali 2024",
                "date": "2024-10-31",
                "type": "Festival",
                "description": "Festival of Lights celebrated by Hindus, Sikhs, and Jains"
            },
            {
                "name": "Black Friday",
                "date": "2024-11-29",
                "type": "Festival",
                "description": "Major shopping day after Thanksgiving"
            },
            {
                "name": "Black Friday 2025",
                "date": "2025-11-28",
                "type": "Festival",
                "description": "Major shopping day after Thanksgiving"
            },
            
            # 2025 Observance Days
            {
                "name": "Mother's Day 2025",
                "date": "2025-03-30",
                "type": "Festival",
                "description": "Mothering Sunday in the UK (4th Sunday of Lent)"
            },
            {
                "name": "Father's Day 2025",
                "date": "2025-06-15",
                "type": "Festival",
                "description": "Day to honor fathers (3rd Sunday in June)"
            },
            {
                "name": "Valentine's Day 2025",
                "date": "2025-02-14",
                "type": "Festival",
                "description": "Day celebrating love and romance"
            },
            {
                "name": "St. Patrick's Day 2025",
                "date": "2025-03-17",
                "type": "Festival",
                "description": "Celebration of Irish culture and heritage"
            },
            {
                "name": "St. George's Day 2025",
                "date": "2025-04-23",
                "type": "Festival",
                "description": "England's national day"
            },
            {
                "name": "Remembrance Day 2025",
                "date": "2025-11-11",
                "type": "Festival",
                "description": "Armistice Day commemorating the end of World War I"
            },
            {
                "name": "Remembrance Sunday 2025",
                "date": "2025-11-09",
                "type": "Festival",
                "description": "Sunday closest to November 11th honoring those who died in the line of duty"
            },
            {
                "name": "Halloween 2025",
                "date": "2025-10-31",
                "type": "Festival",
                "description": "Evening before All Hallows' Day"
            },
            {
                "name": "Pancake Day 2025",
                "date": "2025-03-04",
                "type": "Festival",
                "description": "Shrove Tuesday, the day before Ash Wednesday"
            },
            {
                "name": "Chinese New Year 2025",
                "date": "2025-01-29",
                "type": "Festival",
                "description": "Year of the Snake celebrations"
            },
            {
                "name": "Diwali 2025",
                "date": "2025-10-19",
                "type": "Festival",
                "description": "Festival of Lights celebrated by Hindus, Sikhs, and Jains"
            },

            # 2024-2025 Events Pre and Post Events Days
            {
                "name": "New Year's Eve Impact",
                "date": "2023-12-31",
                "type": "Holiday",
                "description": "Preparation day before New Year's with increased activity"
            },
            {
                "name": "Valentine's Day Preparation",
                "date": "2024-02-13",
                "type": "Festival",
                "description": "Shopping and preparation day before Valentine's Day"
            },
            {
                "name": "Easter Shopping Period",
                "date": "2024-03-29",
                "end_date": "2024-03-30",
                "type": "Holiday",
                "description": "Pre-Easter shopping and preparation days"
            },
            {
                "name": "Tramlines Festival Setup",
                "date": "2024-07-25",
                "type": "Festival",
                "description": "Setup day before Sheffield's biggest city music festival"
            },
            {
                "name": "Tramlines Festival Aftermath",
                "date": "2024-07-29",
                "type": "Festival",
                "description": "Post-festival impact day in Sheffield"
            },
            {
                "name": "Halloween Preparation",
                "date": "2024-10-30",
                "type": "Festival",
                "description": "Preparation day before Halloween"
            },
            {
                "name": "Christmas Shopping Peak",
                "date": "2024-12-23",
                "end_date": "2024-12-24",
                "type": "Holiday",
                "description": "Peak shopping days before Christmas"
            },
            {
                "name": "New Year's Eve Impact",
                "date": "2024-12-31",
                "type": "Holiday",
                "description": "Preparation day before New Year's 2025 with increased activity"
            },
            {
                "name": "Valentine's Day Preparation 2025",
                "date": "2025-02-13",
                "type": "Festival",
                "description": "Shopping and preparation day before Valentine's Day"
            },
            {
                "name": "Easter Shopping Period 2025",
                "date": "2025-04-18",
                "end_date": "2025-04-19",
                "type": "Holiday",
                "description": "Pre-Easter shopping and preparation days"
            },
            {
                "name": "Halloween Preparation 2025",
                "date": "2025-10-30",
                "type": "Festival",
                "description": "Preparation day before Halloween"
            },
            {
                "name": "Christmas Shopping Peak 2025",
                "date": "2025-12-23",
                "end_date": "2025-12-24",
                "type": "Holiday",
                "description": "Peak shopping days before Christmas"
            },

            
        ]
        
        # Filter events by year range
        if end_year is None:
            end_year = start_year + 5
            
        filtered_events = []
        
        # Process all events
        for event in major_events:
            event_year = int(event["date"].split("-")[0])
            
            if start_year <= event_year <= end_year:
                filtered_events.append(event)
                
        data_logger.info(f"Retrieved {len(filtered_events)} major events for years {start_year}-{end_year}")
        return filtered_events
    
    def _create_default_events(self) -> List[Dict[str, Any]]:
        """
        Create default Sheffield events data with real events by combining
        UK holidays and major events
        
        Returns:
            List of dictionaries with default events
        """
        try:
            # Get UK holidays from the holidays library
            uk_holidays = self._get_uk_holidays(2024, 2025)
            
            # Get major events
            major_events = self._get_major_events(2024, 2025)
            
            # Combine holidays and events
            all_events = uk_holidays + major_events
            
            # Sort by date
            all_events.sort(key=lambda x: x["date"])
            
            data_logger.info(f"Created {len(all_events)} default events")
            
            # Save to cache
            self._save_events_data(all_events)
            
            return all_events
            
        except Exception as e:
            data_logger.error(f"Error creating default events: {str(e)}")
            data_logger.error(traceback.format_exc())
            
            # Return a minimal set of events in case of error
            basic_events = [
                {
                    "name": "New Year's Day",
                    "date": "2024-01-01",
                    "type": "Holiday",
                    "description": "Public holiday"
                },
                {
                    "name": "Christmas Day",
                    "date": "2024-12-25",
                    "type": "Holiday",
                    "description": "Public holiday"
                }
            ]
            
            return basic_events
    
    def get_events(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        Get events within a date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of dictionaries with event data
        """
        try:
            events = self._load_events_data()
            
            # If no date range specified, return all events
            if not start_date and not end_date:
                return events
            
            # Parse dates
            start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime(1900, 1, 1)
            end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime(2100, 12, 31)
            
            # Filter events by date
            filtered_events = []
            
            for event in events:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
                
                # Check if event falls within range
                if start <= event_date <= end:
                    filtered_events.append(event)
                
                # Handle multi-day events
                if "end_date" in event:
                    event_end_date = datetime.strptime(event["end_date"], "%Y-%m-%d")
                    
                    # Check if event overlaps with range
                    if (start <= event_end_date and end >= event_date):
                        # Only add if not already added
                        if event not in filtered_events:
                            filtered_events.append(event)
            
            data_logger.info(f"Retrieved {len(filtered_events)} events between {start_date or 'beginning'} and {end_date or 'end'}")
            
            return filtered_events
            
        except Exception as e:
            data_logger.error(f"Error getting events: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []
    
    def add_event(self, event: Dict[str, Any]) -> bool:
        """
        Add a new event
        
        Args:
            event: Dictionary with event data
            
        Returns:
            Success flag
        """
        try:
            # Validate required fields
            required_fields = ["name", "date", "type", "description"]
            missing_fields = [field for field in required_fields if field not in event]
            
            if missing_fields:
                data_logger.error(f"Missing required fields: {missing_fields}")
                return False
            
            # Load existing events
            events = self._load_events_data()
            
            # Add the new event
            events.append(event)
            
            # Save updated events
            success = self._save_events_data(events)
            
            if success:
                data_logger.info(f"Added new event: {event['name']} on {event['date']}")
            
            return success
            
        except Exception as e:
            data_logger.error(f"Error adding event: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def update_event(self, event_name: str, event_date: str, updated_data: Dict[str, Any]) -> bool:
        """
        Update an existing event
        
        Args:
            event_name: Name of the event to update
            event_date: Date of the event to update
            updated_data: Dictionary with updated event data
            
        Returns:
            Success flag
        """
        try:
            # Load existing events
            events = self._load_events_data()
            
            # Find the event to update
            found = False
            
            for i, event in enumerate(events):
                if event["name"] == event_name and event["date"] == event_date:
                    # Update the event
                    events[i].update(updated_data)
                    found = True
                    break
            
            if not found:
                data_logger.error(f"Event not found: {event_name} on {event_date}")
                return False
            
            # Save updated events
            success = self._save_events_data(events)
            
            if success:
                data_logger.info(f"Updated event: {event_name} on {event_date}")
            
            return success
            
        except Exception as e:
            data_logger.error(f"Error updating event: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def delete_event(self, event_name: str, event_date: str) -> bool:
        """
        Delete an existing event
        
        Args:
            event_name: Name of the event to delete
            event_date: Date of the event to delete
            
        Returns:
            Success flag
        """
        try:
            # Load existing events
            events = self._load_events_data()
            
            # Find the event to delete
            original_count = len(events)
            events = [event for event in events if not (event["name"] == event_name and event["date"] == event_date)]
            
            if len(events) == original_count:
                data_logger.error(f"Event not found: {event_name} on {event_date}")
                return False
            
            # Save updated events
            success = self._save_events_data(events)
            
            if success:
                data_logger.info(f"Deleted event: {event_name} on {event_date}")
            
            return success
            
        except Exception as e:
            data_logger.error(f"Error deleting event: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def prepare_events_for_prophet(self, start_date: str, end_date: str, forecast_days: int = 30) -> pd.DataFrame:
        """
        Prepare events data for Prophet model
        
        Args:
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with events data ready for Prophet
        """
        try:
            # Calculate forecast end date
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            forecast_end = end_dt + timedelta(days=forecast_days)
            forecast_end_str = forecast_end.strftime("%Y-%m-%d")
            
            # Get events covering both historical and forecast periods
            events = self.get_events(start_date, forecast_end_str)
            
            if not events:
                data_logger.warning("No events found for the specified period")
                return pd.DataFrame()
            
            # Generate a date range covering the entire period
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            date_range = pd.date_range(start=start_dt, end=forecast_end, freq='D')
            
            # Create a DataFrame with all dates
            df = pd.DataFrame({'ds': date_range})
            
            # Initialize event features
            df['event'] = 0
            #df['event_impact'] = 1.0  # Default multiplier
            df['holiday'] = 0
            df['festival'] = 0
            #df['is_holiday'] = 0  # UK holiday indicator for Prophet
            
            # Process each event
            for event in events:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
                
                # Set flags for the event date
                mask = (df['ds'] == pd.Timestamp(event_date))
                
                if mask.any():
                    df.loc[mask, 'event'] = 1
                    #df.loc[mask, 'event_impact'] = 1.0  # Set to default impact
                    
                    # Set type-specific flags
                    if event["type"] == "Holiday":
                        df.loc[mask, 'holiday'] = 1
                        #df.loc[mask, 'is_holiday'] = 1
                    elif event["type"] == "Festival":
                        df.loc[mask, 'festival'] = 1
                
                # Handle multi-day events
                if "end_date" in event:
                    event_end_date = datetime.strptime(event["end_date"], "%Y-%m-%d")
                    
                    # Create masks for all days in the event
                    event_days = pd.date_range(start=event_date, end=event_end_date, freq='D')
                    for day in event_days:
                        mask = (df['ds'] == day)
                        
                        if mask.any():
                            df.loc[mask, 'event'] = 1
                            #df.loc[mask, 'event_impact'] = 1.0  # Set to default impact
                            
                            # Set type-specific flags
                            if event["type"] == "Holiday":
                                df.loc[mask, 'holiday'] = 1
                                #df.loc[mask, 'is_holiday'] = 1
                            elif event["type"] == "Festival":
                                df.loc[mask, 'festival'] = 1
            
            data_logger.info(f"Prepared events data for Prophet with {len(df)} rows")
            data_logger.info(f"Events in period: {df['event'].sum()}")
            
            return df
            
        except Exception as e:
            data_logger.error(f"Error preparing events for Prophet: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()

# Test function
def test_events_service():
    """Test the EventsService functionality"""
    service = EventsService()
    
    # Test getting all events
    all_events = service.get_events()
    print(f"All events: {len(all_events)}")
    
    # Test getting events in a date range
    range_events = service.get_events('2024-01-01', '2025-12-31')
    print(f"Events in May-Aug 2024: {len(range_events)}")
    for event in range_events:
        print(f"{event['date']}: {event['name']}")
    
    # Test adding an event
    new_event = {
        "name": "Custom Test Event",
        "date": "2024-09-15",
        "type": "Festival",
        "description": "Test event for demonstration"
    }
    
    add_result = service.add_event(new_event)
    print(f"Add event result: {add_result}")
    
    # Test preparation for Prophet
    prophet_data = service.prepare_events_for_prophet('2024-01-01', '2025-12-31', 30)

    # Save the prophet data for 30 Period, 
    prophet_data.to_csv("prophet_data_30_period.csv", index=False)

    # Save the prophet data for 7 Period, 
    prophet_data_7_period = service.prepare_events_for_prophet('2024-01-01', '2025-01-01', 360)
    prophet_data_7_period.to_csv("prophet_data_7_period.csv", index=False)
    

    print(f"Prophet data: {len(prophet_data)} rows")
    print(f"Columns: {prophet_data.columns.tolist()}")
    print(f"Events in period: {prophet_data['event'].sum()}")
    print(f"Sample data:\n{prophet_data[prophet_data['event'] == 1].head()}")
    
    return "Events service test completed successfully"

if __name__ == "__main__":
    test_events_service()