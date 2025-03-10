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

class EventsService:
    """
    Service for retrieving and managing Sheffield event data
    """
    
    def __init__(self, cache_dir: str = None):
        """Initialize the events service"""
        # Cache settings
        self.cache_dir = cache_dir or os.path.join(settings.PROCESSED_DATA_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "events_cache.json")
        
        data_logger.info("EventsService initialized for Sheffield events")
    
    def _load_events_data(self) -> List[Dict[str, Any]]:
        """
        Load events data from the cache file or create default data
        
        Returns:
            List of dictionaries with event data
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            
            # If no cache file exists, create default Sheffield events
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
            with open(self.cache_file, 'w') as f:
                json.dump(events, f, indent=2)
            
            data_logger.info(f"Saved {len(events)} events to cache")
            return True
            
        except Exception as e:
            data_logger.error(f"Error saving events data: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def _create_default_events(self) -> List[Dict[str, Any]]:
        """
        Create default Sheffield events data with real events for 2024
        
        Returns:
            List of dictionaries with default events
        """
        real_events = [
            {
                "name": "New Year's Day",
                "date": "2024-01-01",
                "type": "Holiday",
                "description": "Public holiday"
            },
            {
                "name": "Valentine's Day",
                "date": "2024-02-14",
                "type": "Festival",
                "description": "Special day for couples"
            },
            {
                "name": "St. Patrick's Day",
                "date": "2024-03-17",
                "type": "Holiday",
                "description": "Irish celebration"
            },
            {
                "name": "Good Friday",
                "date": "2024-03-29",
                "type": "Holiday",
                "description": "Public holiday"
            },
            {
                "name": "Easter Sunday",
                "date": "2024-03-31",
                "type": "Holiday",
                "description": "Christian holiday"
            },
            {
                "name": "Easter Monday",
                "date": "2024-04-01",
                "type": "Holiday",
                "description": "Public holiday"
            },
            {
                "name": "Sheffield Beer Week",
                "date": "2024-02-16",
                "end_date": "2024-02-23",
                "type": "Festival",
                "description": "Week-long beer festival across Sheffield"
            },
            {
                "name": "The Crucible Tournament",
                "date": "2024-04-20",
                "end_date": "2024-05-06",
                "type": "Festival",
                "description": "Snooker tournament in Sheffield"
            },
            {
                "name": "Sheffield Half Marathon",
                "date": "2024-04-14",
                "type": "Festival",
                "description": "Half marathon through Sheffield"
            },
            {
                "name": "Early May Bank Holiday",
                "date": "2024-05-06",
                "type": "Holiday",
                "description": "Public holiday"
            },
            {
                "name": "Spring Bank Holiday",
                "date": "2024-05-27",
                "type": "Holiday",
                "description": "Public holiday"
            },
            {
                "name": "Sheffield DocFest",
                "date": "2024-06-11",
                "end_date": "2024-06-16",
                "type": "Festival",
                "description": "Documentary festival in Sheffield"
            },
            {
                "name": "Sheffield Chamber Music Festival",
                "date": "2024-07-12",
                "end_date": "2024-07-20",
                "type": "Festival",
                "description": "Chamber music festival in Sheffield"
            },
            {
                "name": "Tramlines Festival",
                "date": "2024-07-26",
                "end_date": "2024-07-28",
                "type": "Festival",
                "description": "Major music festival in Sheffield"
            },
            {
                "name": "Summer Bank Holiday",
                "date": "2024-08-26",
                "type": "Holiday",
                "description": "Public holiday"
            },
            {
                "name": "Sheffield Food Festival",
                "date": "2024-09-14",
                "end_date": "2024-09-15",
                "type": "Festival",
                "description": "Food festival in Sheffield city center"
            },
            {
                "name": "Sheffield Literary Festival",
                "date": "2024-10-01",
                "end_date": "2024-10-05",
                "type": "Festival",
                "description": "Literary festival in Sheffield"
            },
            {
                "name": "Halloween",
                "date": "2024-10-31",
                "type": "Holiday",
                "description": "Halloween celebrations"
            },
            {
                "name": "Sheffield Comedy Festival",
                "date": "2024-11-01",
                "end_date": "2024-11-10",
                "type": "Festival",
                "description": "Comedy festival in Sheffield"
            },
            {
                "name": "Black Friday",
                "date": "2024-11-29",
                "type": "Festival",
                "description": "Major shopping day"
            },
            {
                "name": "Sheffield Christmas Market",
                "date": "2024-11-14",
                "end_date": "2024-12-24",
                "type": "Festival",
                "description": "Christmas market in Sheffield city center"
            },
            {
                "name": "Christmas Day",
                "date": "2024-12-25",
                "type": "Holiday",
                "description": "Public holiday"
            },
            {
                "name": "Boxing Day",
                "date": "2024-12-26",
                "type": "Holiday",
                "description": "Public holiday"
            },
            {
                "name": "New Year's Eve",
                "date": "2024-12-31",
                "type": "Holiday",
                "description": "New Year's Eve celebrations"
            },
            {
                "name": "Sheffield Pride",
                "date": "2024-06-29",
                "type": "Festival",
                "description": "LGBTIQ+ celebration in Sheffield"
            }
        ]
        
        data_logger.info(f"Created {len(real_events)} real Sheffield events for 2024")
        
        # Save to cache
        self._save_events_data(real_events)
        
        return real_events
    
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
            df['event_impact'] = 1.0  # Default multiplier
            df['holiday'] = 0
            df['festival'] = 0
            df['is_holiday'] = 0  # UK holiday indicator for Prophet
            
            # Process each event
            for event in events:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
                
                # Set flags for the event date
                mask = (df['ds'] == pd.Timestamp(event_date))
                
                if mask.any():
                    df.loc[mask, 'event'] = 1
                    df.loc[mask, 'event_impact'] = 1.0  # Set to default impact
                    
                    # Set type-specific flags
                    if event["type"] == "Holiday":
                        df.loc[mask, 'holiday'] = 1
                        df.loc[mask, 'is_holiday'] = 1
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
                            df.loc[mask, 'event_impact'] = 1.0  # Set to default impact
                            
                            # Set type-specific flags
                            if event["type"] == "Holiday":
                                df.loc[mask, 'holiday'] = 1
                                df.loc[mask, 'is_holiday'] = 1
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
    range_events = service.get_events('2024-05-01', '2024-08-31')
    print(f"Events in May-Aug 2024: {len(range_events)}")
    for event in range_events:
        print(f"{event['date']}: {event['name']}")
    
    # Test adding an event
    new_event = {
        "name": "Sheffield Half Marathon",
        "date": "2024-04-14",
        "type": "Festival",
        "description": "Half marathon through Sheffield"
    }
    
    add_result = service.add_event(new_event)
    print(f"Add event result: {add_result}")
    
    # Test preparation for Prophet
    prophet_data = service.prepare_events_for_prophet('2024-01-01', '2024-02-29', 30)
    print(f"Prophet data: {len(prophet_data)} rows")
    print(f"Columns: {prophet_data.columns.tolist()}")
    print(f"Events in period: {prophet_data['event'].sum()}")
    print(f"Sample data:\n{prophet_data[prophet_data['event'] == 1].head()}")
    
    return "Events service test completed successfully"

if __name__ == "__main__":
    test_events_service()