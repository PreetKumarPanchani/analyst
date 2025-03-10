# app/api/routes/events.py
from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any, Optional
import traceback
from datetime import datetime, timedelta

from app.core.logger import api_logger
from app.services.events_service import EventsService
from app.services.weather_service import WeatherService

router = APIRouter()
events_service = EventsService()
weather_service = WeatherService()

@router.get("/events", response_model=List[Dict[str, Any]])
async def get_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get events within a date range"""
    try:
        api_logger.info(f"Request for events from {start_date} to {end_date}")
        
        events = events_service.get_events(start_date, end_date)
        return events
    except Exception as e:
        api_logger.error(f"Error getting events: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/events", response_model=Dict[str, Any])
async def add_event(event: Dict[str, Any] = Body(...)):
    """Add a new event"""
    try:
        api_logger.info(f"Request to add new event: {event.get('name')}")
        
        success = events_service.add_event(event)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add event")
            
        return {
            "success": True,
            "message": f"Successfully added event: {event.get('name')}"
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error adding event: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/events/{event_name}", response_model=Dict[str, Any])
async def update_event(
    event_name: str,
    event_date: str,
    updated_data: Dict[str, Any] = Body(...)
):
    """Update an existing event"""
    try:
        api_logger.info(f"Request to update event: {event_name} on {event_date}")
        
        success = events_service.update_event(event_name, event_date, updated_data)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Event not found: {event_name} on {event_date}")
            
        return {
            "success": True,
            "message": f"Successfully updated event: {event_name} on {event_date}"
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error updating event: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/events/{event_name}", response_model=Dict[str, Any])
async def delete_event(
    event_name: str,
    event_date: str
):
    """Delete an existing event"""
    try:
        api_logger.info(f"Request to delete event: {event_name} on {event_date}")
        
        success = events_service.delete_event(event_name, event_date)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Event not found: {event_name} on {event_date}")
            
        return {
            "success": True,
            "message": f"Successfully deleted event: {event_name} on {event_date}"
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error deleting event: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weather/current", response_model=Dict[str, Any])
async def get_current_weather():
    """Get current weather in Sheffield"""
    try:
        api_logger.info("Request for current weather")
        
        weather_data = weather_service.get_current_weather()
        
        if not weather_data:
            raise HTTPException(status_code=500, detail="Failed to get current weather data")
            
        return weather_data
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting current weather: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weather/forecast", response_model=List[Dict[str, Any]])
async def get_weather_forecast(
    days: int = 7
):
    """Get weather forecast for Sheffield"""
    try:
        api_logger.info(f"Request for {days}-day weather forecast")
        
        forecast = weather_service.get_forecast(days)
        
        if not forecast:
            raise HTTPException(status_code=500, detail="Failed to get weather forecast")
            
        return forecast
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting weather forecast: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/weather/historical", response_model=List[Dict[str, Any]])
async def get_historical_weather(
    start_date: str,
    end_date: str
):
    """Get historical weather data for Sheffield"""
    try:
        api_logger.info(f"Request for historical weather from {start_date} to {end_date}")
        
        # Validate dates
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        historical = weather_service.get_historical_weather(start_date, end_date)
        
        if not historical:
            raise HTTPException(status_code=500, detail="Failed to get historical weather data")
            
        return historical
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting historical weather: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))