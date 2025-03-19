# app/api/routes/forecasts.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import traceback

from app.core.logger import api_logger
from app.services.forecast_service import ForecastService

router = APIRouter()
forecast_service = ForecastService()

@router.get("/revenue/{company}", response_model=Dict[str, Any])
async def forecast_revenue(
    company: str,
    periods: int = Query(15, ge=1, le=365, description="Number of days to forecast"),
    include_weather: bool = Query(True, description="Include weather data"),
    include_events: bool = Query(True, description="Include events data"),
    force_retrain: bool = Query(False, description="Force model retraining")
):
    """Generate revenue forecast for a company"""
    try:
        api_logger.info(f"Request for revenue forecast for company: {company}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
            
        forecast = forecast_service.forecast_company_revenue(
            company,
            periods=periods,
            include_weather=include_weather,
            include_events=include_events,
            force_retrain=force_retrain
        )
        
        if not forecast.get("success", False):
            raise HTTPException(status_code=500, detail=forecast.get("error", "Failed to generate forecast"))
            
        return forecast
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error generating revenue forecast: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/category/{company}/{category}", response_model=Dict[str, Any])
async def forecast_category(
    company: str,
    category: str,
    periods: int = Query(15, ge=1, le=365, description="Number of days to forecast"),
    include_weather: bool = Query(True, description="Include weather data"),
    include_events: bool = Query(True, description="Include events data"),
    force_retrain: bool = Query(False, description="Force model retraining")
):
    """Generate sales forecast for a product category"""
    try:
        api_logger.info(f"Request for category forecast for company: {company}, category: {category}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
            
        forecast = forecast_service.forecast_category_sales(
            company,
            category,
            periods=periods,
            include_weather=include_weather,
            include_events=include_events,
            force_retrain=force_retrain
        )
        
        if not forecast.get("success", False):
            raise HTTPException(status_code=500, detail=forecast.get("error", "Failed to generate forecast"))
            
        return forecast
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error generating category forecast: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/product/{company}/{product}", response_model=Dict[str, Any])
async def forecast_product(
    company: str,
    product: str,
    periods: int = Query(15, ge=1, le=365, description="Number of days to forecast"),
    include_weather: bool = Query(True, description="Include weather data"),
    include_events: bool = Query(True, description="Include events data"),
    force_retrain: bool = Query(False, description="Force model retraining")
):
    """Generate sales forecast for a specific product"""
    try:
        api_logger.info(f"Request for product forecast for company: {company}, product: {product}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
            
        forecast = forecast_service.forecast_product_sales(
            company,
            product,
            periods=periods,
            include_weather=include_weather,
            include_events=include_events,
            force_retrain=force_retrain
        )
        
        if not forecast.get("success", False):
            raise HTTPException(status_code=500, detail=forecast.get("error", "Failed to generate forecast"))
            
        return forecast
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error generating product forecast: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
