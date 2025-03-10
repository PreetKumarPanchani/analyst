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
    periods: int = Query(30, ge=1, le=365, description="Number of days to forecast"),
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
    periods: int = Query(30, ge=1, le=365, description="Number of days to forecast"),
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
    periods: int = Query(30, ge=1, le=365, description="Number of days to forecast"),
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

@router.get("/insights/{company}", response_model=Dict[str, Any])
async def get_forecast_insights(company: str):
    """Get insights from forecasts for a company"""
    try:
        api_logger.info(f"Request for forecast insights for company: {company}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
            
        # Get revenue forecast
        revenue_forecast = forecast_service.forecast_company_revenue(company)
        
        if not revenue_forecast.get("success", False):
            raise HTTPException(status_code=500, detail="Failed to generate revenue forecast")
            
        # Get top products
        top_products = forecast_service.get_top_products(company, limit=5)
        
        # Get forecasts for top products
        product_forecasts = {}
        for product in top_products:
            product_name = product["product"]
            forecast = forecast_service.forecast_product_sales(company, product_name)
            if forecast.get("success", False):
                product_forecasts[product_name] = forecast
        
        # Generate insights
        insights = {
            "revenue_trend": "increasing" if revenue_forecast["predictions"][-1] > revenue_forecast["predictions"][0] else "decreasing",
            "forecast_period": {
                "start": revenue_forecast["dates"][len(revenue_forecast["actuals"])],
                "end": revenue_forecast["dates"][-1]
            },
            "expected_revenue": sum(revenue_forecast["predictions"][len(revenue_forecast["actuals"]):]),
            "top_products": top_products,
            "product_insights": []
        }
        
        # Add product-specific insights
        for product in top_products:
            product_name = product["product"]
            if product_name in product_forecasts:
                forecast = product_forecasts[product_name]
                
                # Calculate growth rate
                forecast_start = len(forecast["actuals"])
                forecast_values = forecast["predictions"][forecast_start:]
                
                if forecast_values[0] > 0:
                    growth_rate = (forecast_values[-1] - forecast_values[0]) / forecast_values[0] * 100
                else:
                    growth_rate = 0
                
                insights["product_insights"].append({
                    "product": product_name,
                    "expected_quantity": sum(forecast_values),
                    "growth_rate": growth_rate,
                    "trend": "increasing" if growth_rate > 0 else "decreasing" if growth_rate < 0 else "stable"
                })
        
        return {
            "success": True,
            "company": company,
            "insights": insights
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error generating forecast insights: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))