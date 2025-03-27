# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import traceback

from app.api.routes import sales, forecasts, events, uploads
from app.core.config import settings
from app.core.logger import logger, api_logger
from app.data.loader import DataLoader
from app.data.processor import DataProcessor
from app.services.s3_service import S3Service

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API routes
app.include_router(sales.router, prefix=f"{settings.API_V1_STR}/sales", tags=["sales"])
app.include_router(forecasts.router, prefix=f"{settings.API_V1_STR}/forecasts", tags=["forecasts"])
app.include_router(events.router, prefix=f"{settings.API_V1_STR}/external", tags=["external"])
app.include_router(uploads.router, prefix=f"{settings.API_V1_STR}/uploads", tags=["uploads"])

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Log the request
    api_logger.info(f"Request {request.method} {request.url.path}")
    
    # Process the request
    try:
        response = await call_next(request)
        
        # Log the response
        process_time = time.time() - start_time
        api_logger.info(f"Response {request.method} {request.url.path} completed in {process_time:.3f}s with status {response.status_code}")
        
        return response
        
    except Exception as e:
        # Log the error
        api_logger.error(f"Error processing request {request.method} {request.url.path}: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise

'''
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info(f"Starting {settings.PROJECT_NAME}")
    
    # Create directories
    settings.create_directories()
    
    # Pre-process data if needed
    try:
        loader = DataLoader()
        processor = DataProcessor()
        
        # Process data for both companies
        for company in ["forge", "cpl"]:
            # Check if processed data already exists
            processed_data = processor.load_processed_data(company)
            
            if not processed_data:
                logger.info(f"No processed data found for {company}. Processing raw data...")
                
                # Load raw data
                raw_data = loader.load_company_data(company)
                
                if raw_data:
                    # Process data
                    processor.process_company_data(company, raw_data)
                    logger.info(f"Successfully processed data for {company}")
                else:
                    logger.warning(f"No raw data found for {company}")
    
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error(traceback.format_exc())


'''

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Create required directories
        settings.create_directories()
        
        # Initialize S3 service
        s3_service = S3Service()
        if settings.USE_S3_STORAGE:
            api_logger.info(f"Initialized S3 storage with bucket: {settings.AWS_S3_BUCKET_NAME}")
        else:
            api_logger.info("Using local file storage (S3 storage disabled)")
        
        # Initialize data processor with test data
        api_logger.info("Testing data processing pipeline...")
        try:
            loader = DataLoader()
            processor = DataProcessor()
            
            # Process data for both companies
            for company in ["forge", "cpl"]:
                # Check if processed data already exists
                processed_data = processor.load_processed_data(company)
                
                if not processed_data:
                    logger.info(f"No processed data found for {company}. Processing raw data...")
                    
                    # Load raw data
                    raw_data = loader.load_company_data(company)
                    
                    if raw_data:
                        # Process data
                        processor.process_company_data(company, raw_data)
                        logger.info(f"Successfully processed data for {company}")
                    else:
                        logger.warning(f"No raw data found for {company}")
        
        except Exception as e:
            api_logger.error(f"Data processor test failed: {str(e)}")
        
        # Log startup
        api_logger.info("Application started successfully")
        
    except Exception as e:
        api_logger.error(f"Startup error: {str(e)}")
        api_logger.error(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info(f"Shutting down {settings.PROJECT_NAME}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.PROJECT_NAME,
        "version": "0.1.0",
        "description": "API for sales forecasting in Sheffield",
        "docs": f"{settings.API_V1_STR}/docs"
    }

@app.get(f"{settings.API_V1_STR}/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)