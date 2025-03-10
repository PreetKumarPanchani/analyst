# app/api/routes/sales.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import traceback

from app.core.logger import api_logger
from app.data.loader import DataLoader
from app.data.processor import DataProcessor
from app.services.forecast_service import ForecastService

router = APIRouter()
loader = DataLoader()
processor = DataProcessor()
forecast_service = ForecastService()

@router.get("/companies", response_model=List[str])
async def get_companies():
    """Get list of available companies"""
    try:
        api_logger.info("Request for list of companies")
        return ["forge", "cpl"]
    except Exception as e:
        api_logger.error(f"Error getting companies: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categories/{company}", response_model=List[str])
async def get_categories(company: str):
    """Get list of product categories for a company"""
    try:
        api_logger.info(f"Request for categories for company: {company}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
            
        categories = forecast_service.get_categories(company)
        return categories
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting categories: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/products/{company}", response_model=List[str])
async def get_products(
    company: str,
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get list of products for a company"""
    try:
        api_logger.info(f"Request for products for company: {company}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
            
        products = forecast_service.get_products(company, category)
        return products
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting products: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top-products/{company}", response_model=List[Dict[str, Any]])
async def get_top_products(
    company: str,
    limit: int = Query(10, ge=1, le=100, description="Number of products to return")
):
    """Get top products by sales volume"""
    try:
        api_logger.info(f"Request for top products for company: {company}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
            
        top_products = forecast_service.get_top_products(company, limit)
        return top_products
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting top products: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/{company}")
async def process_company_data(company: str):
    """Process raw data for a company"""
    try:
        api_logger.info(f"Request to process data for company: {company}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
            
        # Load raw data
        raw_data = loader.load_company_data(company)
        
        if not raw_data:
            raise HTTPException(status_code=404, detail=f"No raw data found for company: {company}")
            
        # Process data
        processed_data = processor.process_company_data(company, raw_data)
        
        if not processed_data:
            raise HTTPException(status_code=500, detail=f"Failed to process data for company: {company}")
            
        return {
            "success": True,
            "message": f"Successfully processed data for company: {company}",
            "data_types": list(processed_data.keys())
        }
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error processing company data: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))