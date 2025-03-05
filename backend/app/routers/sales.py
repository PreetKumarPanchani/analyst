from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from app.services.data_service import DataService

router = APIRouter()

# Dependency to get the data service
def get_data_service():
    service = DataService("./data")
    service.load_data()
    return service

@router.get("/summary")
async def get_sales_summary(
    company: str = Query(..., description="Company name (forge or cpl)"),
    data_service: DataService = Depends(get_data_service)
):
    """
    Get summary statistics for sales.
    """
    try:
        summary = data_service.get_sales_summary(company)
        return summary
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sales summary: {str(e)}")

@router.get("/monthly")
async def get_monthly_sales(
    company: str = Query(..., description="Company name (forge or cpl)"),
    data_service: DataService = Depends(get_data_service)
):
    """
    Get monthly sales statistics.
    """
    try:
        monthly_sales = data_service.get_monthly_sales(company)
        return monthly_sales
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting monthly sales: {str(e)}")

@router.get("/daily")
async def get_daily_sales(
    company: str = Query(..., description="Company name (forge or cpl)"),
    days: int = Query(30, description="Number of days to include"),
    data_service: DataService = Depends(get_data_service)
):
    """
    Get daily sales statistics for the last N days.
    """
    try:
        daily_sales = data_service.get_daily_sales(company, days)
        return daily_sales
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting daily sales: {str(e)}")

@router.get("/registers")
async def get_register_summary(
    company: str = Query(..., description="Company name (forge or cpl)"),
    data_service: DataService = Depends(get_data_service)
):
    """
    Get sales summary by register.
    """
    try:
        register_summary = data_service.get_register_summary(company)
        return register_summary
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting register summary: {str(e)}")

@router.get("/top-products")
async def get_top_products(
    company: str = Query(..., description="Company name (forge or cpl)"),
    limit: int = Query(10, description="Number of products to return"),
    data_service: DataService = Depends(get_data_service)
):
    """
    Get the top N products by sales.
    """
    try:
        top_products = data_service.get_top_products(company, limit)
        return top_products
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting top products: {str(e)}")

@router.get("/categories")
async def get_category_summary(
    company: str = Query(..., description="Company name (forge or cpl)"),
    data_service: DataService = Depends(get_data_service)
):
    """
    Get sales summary by category.
    """
    try:
        category_summary = data_service.get_category_summary(company)
        return category_summary
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting category summary: {str(e)}")