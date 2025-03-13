# app/api/routes/forecasts.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
import traceback

from app.core.logger import api_logger
from app.services.forecast_service import ForecastService

router = APIRouter()
forecast_service = ForecastService()

@router.get("/models", response_model=List[str])
async def get_available_models():
    """Get available forecasting models"""
    try:
        models = forecast_service.get_available_model_types()
        api_logger.info(f"Available models: {models}")
        return models
    except Exception as e:
        api_logger.error(f"Error getting available models: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/revenue/{company}", response_model=Dict[str, Any])
async def forecast_revenue(
    company: str,
    periods: int = Query(7, ge=1, le=365, description="Number of days to forecast"),
    include_weather: bool = Query(True, description="Include weather data"),
    include_events: bool = Query(True, description="Include events data"),
    force_retrain: bool = Query(False, description="Force model retraining"),
    model_type: str = Query("prophet", description="Forecasting model to use (prophet, timegpt)"),
    finetune: bool = Query(True, description="Fine-tune TimeGPT model (only for TimeGPT)"),
    finetune_steps: int = Query(10, ge=1, le=100, description="Number of fine-tuning steps (only for TimeGPT)"),
    finetune_loss: str = Query("mse", description="Loss function for fine-tuning (only for TimeGPT)"),
    finetune_depth: int = Query(2, ge=1, le=5, description="Depth for fine-tuning (only for TimeGPT)")
):
    """Generate revenue forecast for a company"""
    try:
        api_logger.info(f"Request for revenue forecast for company: {company} using model: {model_type}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
        
        # Check if requested model is available
        available_models = forecast_service.get_available_model_types()
        if model_type.lower() not in [m.lower() for m in available_models]:
            raise HTTPException(status_code=400, detail=f"Model '{model_type}' not available. Use one of: {available_models}")
            
        forecast = forecast_service.forecast_company_revenue(
            company,
            periods=periods,
            include_weather=include_weather,
            include_events=include_events,
            force_retrain=force_retrain,
            model_type=model_type,
            finetune=finetune,
            finetune_steps=finetune_steps,
            finetune_loss=finetune_loss,
            finetune_depth=finetune_depth
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
    periods: int = Query(7, ge=1, le=365, description="Number of days to forecast"),
    include_weather: bool = Query(True, description="Include weather data"),
    include_events: bool = Query(True, description="Include events data"),
    force_retrain: bool = Query(False, description="Force model retraining"),
    model_type: str = Query("prophet", description="Forecasting model to use (prophet, timegpt)"),
    finetune: bool = Query(True, description="Fine-tune TimeGPT model (only for TimeGPT)"),
    finetune_steps: int = Query(10, ge=1, le=100, description="Number of fine-tuning steps (only for TimeGPT)"),
    finetune_loss: str = Query("mse", description="Loss function for fine-tuning (only for TimeGPT)"),
    finetune_depth: int = Query(2, ge=1, le=5, description="Depth for fine-tuning (only for TimeGPT)")
):
    """Generate sales forecast for a product category"""
    try:
        api_logger.info(f"Request for category forecast for company: {company}, category: {category} using model: {model_type}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
        
        # Check if requested model is available
        available_models = forecast_service.get_available_model_types()
        if model_type.lower() not in [m.lower() for m in available_models]:
            raise HTTPException(status_code=400, detail=f"Model '{model_type}' not available. Use one of: {available_models}")
            
        forecast = forecast_service.forecast_category_sales(
            company,
            category,
            periods=periods,
            include_weather=include_weather,
            include_events=include_events,
            force_retrain=force_retrain,
            model_type=model_type,
            finetune=finetune,
            finetune_steps=finetune_steps,
            finetune_loss=finetune_loss,
            finetune_depth=finetune_depth
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
    periods: int = Query(7, ge=1, le=365, description="Number of days to forecast"),
    include_weather: bool = Query(True, description="Include weather data"),
    include_events: bool = Query(True, description="Include events data"),
    force_retrain: bool = Query(False, description="Force model retraining"),
    model_type: str = Query("prophet", description="Forecasting model to use (prophet, timegpt)"),
    finetune: bool = Query(True, description="Fine-tune TimeGPT model (only for TimeGPT)"),
    finetune_steps: int = Query(10, ge=1, le=100, description="Number of fine-tuning steps (only for TimeGPT)"),
    finetune_loss: str = Query("mse", description="Loss function for fine-tuning (only for TimeGPT)"),
    finetune_depth: int = Query(2, ge=1, le=5, description="Depth for fine-tuning (only for TimeGPT)")
):
    """Generate sales forecast for a specific product"""
    try:
        api_logger.info(f"Request for product forecast for company: {company}, product: {product} using model: {model_type}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
        
        # Check if requested model is available
        available_models = forecast_service.get_available_model_types()
        if model_type.lower() not in [m.lower() for m in available_models]:
            raise HTTPException(status_code=400, detail=f"Model '{model_type}' not available. Use one of: {available_models}")
            
        forecast = forecast_service.forecast_product_sales(
            company,
            product,
            periods=periods,
            include_weather=include_weather,
            include_events=include_events,
            force_retrain=force_retrain,
            model_type=model_type,
            finetune=finetune,
            finetune_steps=finetune_steps,
            finetune_loss=finetune_loss,
            finetune_depth=finetune_depth
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
async def get_forecast_insights(
    company: str,
    model_type: str = Query("prophet", description="Forecasting model to use (prophet, timegpt)")
):
    """Get insights from forecasts for a company"""
    try:
        api_logger.info(f"Request for forecast insights for company: {company} using model: {model_type}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
        
        # Check if requested model is available
        available_models = forecast_service.get_available_model_types()
        if model_type.lower() not in [m.lower() for m in available_models]:
            raise HTTPException(status_code=400, detail=f"Model '{model_type}' not available. Use one of: {available_models}")
            
        # Get revenue forecast
        revenue_forecast = forecast_service.forecast_company_revenue(company, model_type=model_type)
        
        if not revenue_forecast.get("success", False):
            raise HTTPException(status_code=500, detail="Failed to generate revenue forecast")
            
        # Get top products
        top_products = forecast_service.get_top_products(company, limit=5)
        
        # Get forecasts for top products
        product_forecasts = {}
        for product in top_products:
            product_name = product["product"]
            forecast = forecast_service.forecast_product_sales(company, product_name, model_type=model_type)
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
            "product_insights": [],
            "model_type": model_type
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

@router.get("/compare/{company}", response_model=Dict[str, Any])
async def compare_models(
    company: str,
    periods: int = Query(7, ge=1, le=365, description="Number of days to forecast")
):
    """Compare different forecasting models for a company's revenue"""
    try:
        api_logger.info(f"Request to compare models for company: {company}")
        
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=404, detail=f"Company not found: {company}")
            
        # Get available models
        available_models = forecast_service.get_available_model_types()
        
        if len(available_models) < 2:
            return {
                "success": True,
                "message": "Only one model available, no comparison possible",
                "available_models": available_models
            }
            
        # Generate forecasts with each model
        forecasts = {}
        metrics = {}
        
        for model_type in available_models:
            forecast = forecast_service.forecast_company_revenue(
                company, 
                periods=periods,
                model_type=model_type
            )
            
            if forecast.get("success", False):
                forecasts[model_type] = forecast
                metrics[model_type] = forecast.get("metrics", {})
        
        # Create comparison result
        comparison = {
            "success": True,
            "company": company,
            "periods": periods,
            "available_models": available_models,
            "metrics": metrics,
            "dates": forecasts[available_models[0]]["dates"] if available_models else [],
            "actuals": forecasts[available_models[0]]["actuals"] if available_models else [],
            "predictions": {
                model_type: forecast["predictions"] for model_type, forecast in forecasts.items()
            },
            "lower_bounds": {
                model_type: forecast["lower_bound"] for model_type, forecast in forecasts.items()
            },
            "upper_bounds": {
                model_type: forecast["upper_bound"] for model_type, forecast in forecasts.items()
            }
        }
        
        # Determine best model based on MAPE (if available)
        best_model = None
        best_mape = float('inf')
        
        for model_type, model_metrics in metrics.items():
            mape = model_metrics.get("mape")
            if mape is not None and mape < best_mape:
                best_mape = mape
                best_model = model_type
        
        comparison["best_model"] = best_model
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error comparing models: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))