from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any
from app.services.data_service import DataService
from app.services.forecast import ForecastService
from app.models.schemas import ForecastRequest

router = APIRouter()

# Dependency to get the forecast service
def get_forecast_service():
    data_service = DataService("./data")
    data_service.load_data()
    return ForecastService(data_service)

@router.get("/sales")
async def forecast_sales(
    company: str = Query(..., description="Company name (forge or cpl)"),
    periods: int = Query(30, description="Number of days to forecast"),
    forecast_service: ForecastService = Depends(get_forecast_service)
):
    """
    Forecast sales for the specified company.
    """
    try:
        forecast_results = forecast_service.forecast_sales(company, periods)
        return forecast_results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error forecasting sales: {str(e)}")

@router.get("/components")
async def get_forecast_components(
    company: str = Query(..., description="Company name (forge or cpl)"),
    forecast_service: ForecastService = Depends(get_forecast_service)
):
    """
    Get the forecast components (trend, weekly seasonality, yearly seasonality).
    """
    try:
        components = forecast_service.get_forecast_components(company)
        return components
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting forecast components: {str(e)}")