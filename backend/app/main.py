from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import sales, forecast

app = FastAPI(
    title="Bakery Analytics API",
    description="API for analyzing bakery sales data and forecasting",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sales.router, prefix="/api/sales", tags=["sales"])
app.include_router(forecast.router, prefix="/api/forecast", tags=["forecast"])

@app.get("/")
async def root():
    return {"message": "Welcome to Bakery Analytics API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)