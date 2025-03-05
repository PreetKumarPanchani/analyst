from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import date, datetime

class SaleItem(BaseModel):
    product_id: str
    product_name: str
    category: str
    quantity: float
    unit_price: float
    total: float
    
class Sale(BaseModel):
    sale_id: str
    sale_date: date
    sale_time: str
    register: str
    staff: str
    quantity: int
    total: float
    items: List[SaleItem] = []
    
class SalesSummary(BaseModel):
    total_sales: float
    transaction_count: int
    avg_basket: float
    start_date: date
    end_date: date
    
class CategorySummary(BaseModel):
    category: str
    sales: float
    quantity: int
    product_count: int

class ProductSummary(BaseModel):
    product_name: str
    category: str
    sales: float
    quantity: int
    
class MonthlySales(BaseModel):
    month: str
    sales: float
    transactions: int
    avg_basket: float
    
class DailySales(BaseModel):
    date: date
    sales: float
    transactions: int
    avg_basket: float
    
class RegisterSummary(BaseModel):
    register: str
    sales: float
    transactions: int
    avg_basket: float
    
class ForecastRequest(BaseModel):
    company: str = Field(..., description="Company name (forge or cpl)")
    periods: int = Field(30, description="Number of days to forecast")
    
class ForecastPoint(BaseModel):
    date: date
    sales: float
    sales_lower: float
    sales_upper: float
    
class ForecastResponse(BaseModel):
    forecast: List[ForecastPoint]
    model_metrics: Dict[str, float]