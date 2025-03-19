# app/data/processor.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback
import sys
#sys.path.append(os.path.dirname((os.path.abspath(__file__))))
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from app.core.logger import data_logger
from app.core.config import settings
from app.services.s3_service import S3Service

class DataProcessor:
    """
    Service for processing sales data for analysis and forecasting
    """
    
    def __init__(self, processed_dir: str = "data/processed"):
        """Initialize the data processor"""
        self.processed_dir = processed_dir
        self.s3_service = S3Service()
        self.use_s3 = settings.USE_S3_STORAGE
        
        if not self.use_s3:
            os.makedirs(self.processed_dir, exist_ok=True)
            
        data_logger.info(f"DataProcessor initialized with output directory: {processed_dir}")
    
    def process_company_data(self, company: str, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all data for a specific company for analysis and forecasting
        
        Args:
            company: Company name (forge or cpl)
            data: Dictionary of raw DataFrames
            
        Returns:
            Dictionary of processed DataFrames
        """
        try:
            data_logger.info(f"Processing data for company: {company}")
            
            result = {}
            
            # Process daily sales
            result["daily_sales"] = self._process_daily_sales(company, data["sales"])
            
            # Process sales by category
            result["category_sales"] = self._process_category_sales(company, data["sales_items"])
            
            # Process sales by product
            result["product_sales"] = self._process_product_sales(company, data["sales_items"])
            
            # Process payment methods
            result["payment_methods"] = self._process_payment_methods(company, data["sales_payments"])
            

            # To generate the product-category mapping
            self.generate_product_category_mapping(company, data["sales_items"])
            
            # Save processed data
            self._save_processed_data(company, result)
            
            data_logger.info(f"Successfully processed data for company: {company}")
            return result
            
        except Exception as e:
            data_logger.error(f"Error processing company data for {company}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {}
    
    def _process_daily_sales(self, company: str, sales_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process daily sales data
        
        Args:
            company: Company name
            sales_df: Sales DataFrame
            
        Returns:
            Processed daily sales DataFrame
        """
        try:
            # Convert sale date to datetime
            sales_df["Sale Date"] = pd.to_datetime(sales_df["Sale Date"])
            
            # Group by date and compute daily metrics
            daily_sales = sales_df.groupby(sales_df["Sale Date"].dt.date).agg({
                "Sale ID": "count",
                "Total": "sum",
                "Quantity": "sum"
            }).reset_index()
            
            # Rename columns
            daily_sales.columns = ["date", "transaction_count", "total_revenue", "total_quantity"]
            
            # Add company column
            daily_sales["company"] = company
            
            # Ensure date is datetime
            daily_sales["date"] = pd.to_datetime(daily_sales["date"])
            
            # Add derived time features
            daily_sales['dayofweek'] = daily_sales['date'].dt.dayofweek
            daily_sales['month'] = daily_sales['date'].dt.month
            daily_sales['year'] = daily_sales['date'].dt.year
            daily_sales['is_weekend'] = (daily_sales['dayofweek'] >= 5).astype(int)
            
            data_logger.info(f"Processed daily sales for {company}: {len(daily_sales)} rows")
            return daily_sales
            
        except Exception as e:
            data_logger.error(f"Error processing daily sales: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _process_category_sales(self, company: str, sales_items_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process sales by category
        
        Args:
            company: Company name
            sales_items_df: Sales items DataFrame
            
        Returns:
            Processed category sales DataFrame
        """
        try:
            # Convert sale date to datetime
            sales_items_df["Sale Date"] = pd.to_datetime(sales_items_df["Sale Date"])
            
            # Group by date and category
            category_sales = sales_items_df.groupby([
                sales_items_df["Sale Date"].dt.date,
                "Category"
            ]).agg({
                "Total": "sum",
                "Quantity": "sum"
            }).reset_index()
            
            # Rename columns
            category_sales.columns = ["date", "category", "revenue", "quantity"]
            
            # Add company column
            category_sales["company"] = company
            
            # Ensure date is datetime
            category_sales["date"] = pd.to_datetime(category_sales["date"])
            
            # Add derived time features
            category_sales['dayofweek'] = category_sales['date'].dt.dayofweek
            category_sales['month'] = category_sales['date'].dt.month
            category_sales['year'] = category_sales['date'].dt.year
            category_sales['is_weekend'] = (category_sales['dayofweek'] >= 5).astype(int)
            
            data_logger.info(f"Processed category sales for {company}: {len(category_sales)} rows")
            return category_sales
            
        except Exception as e:
            data_logger.error(f"Error processing category sales: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
        
    def generate_product_category_mapping(self, company: str, sales_items_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a mapping between products and their categories
        
        Args:
            company: Company name
            sales_items_df: Sales items DataFrame
            
        Returns:
            DataFrame with product-category mapping
        """
        try:
            # Get unique product-category pairs
            product_category = sales_items_df[["Product Name", "Category"]].drop_duplicates()
            
            # Rename columns
            product_category.columns = ["product", "category"]
            
            # Add company column
            product_category["company"] = company
            
            # Save the mapping
            if self.use_s3:
                # Save to S3
                s3_key = f"{settings.S3_PROCESSED_PREFIX}{company}/product_category_map.csv"
                self.s3_service.write_csv_file(product_category, s3_key)
            else:
                # Save to local filesystem
                mapping_dir = os.path.join(self.processed_dir, company)
                os.makedirs(mapping_dir, exist_ok=True) 
                file_path = os.path.join(mapping_dir, "product_category_map.csv")
                product_category.to_csv(file_path, index=False)
            
            data_logger.info(f"Generated product-category mapping for {company}: {len(product_category)} products")
            return product_category
            
        except Exception as e:
            data_logger.error(f"Error generating product-category mapping: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    def _process_product_sales(self, company: str, sales_items_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process sales by product
        
        Args:
            company: Company name
            sales_items_df: Sales items DataFrame
            
        Returns:
            Processed product sales DataFrame
        """
        try:
            # Convert sale date to datetime
            sales_items_df["Sale Date"] = pd.to_datetime(sales_items_df["Sale Date"])
            
            # Group by date and product
            product_sales = sales_items_df.groupby([
                sales_items_df["Sale Date"].dt.date,
                "Product Name"
            ]).agg({
                "Total": "sum",
                "Quantity": "sum"
            }).reset_index()
            
            # Rename columns
            product_sales.columns = ["date", "product", "revenue", "quantity"]
            
            # Add company column
            product_sales["company"] = company
            
            # Ensure date is datetime
            product_sales["date"] = pd.to_datetime(product_sales["date"])
            
            # Add derived time features
            product_sales['dayofweek'] = product_sales['date'].dt.dayofweek
            product_sales['month'] = product_sales['date'].dt.month
            product_sales['year'] = product_sales['date'].dt.year
            product_sales['is_weekend'] = (product_sales['dayofweek'] >= 5).astype(int)
            
            data_logger.info(f"Processed product sales for {company}: {len(product_sales)} rows")
            return product_sales
            
        except Exception as e:
            data_logger.error(f"Error processing product sales: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _process_payment_methods(self, company: str, sales_payments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process payment methods
        
        Args:
            company: Company name
            sales_payments_df: Sales payments DataFrame
            
        Returns:
            Processed payment methods DataFrame
        """
        try:
            # Convert payment date to datetime
            sales_payments_df["Payment Date"] = pd.to_datetime(sales_payments_df["Payment Date"])
            
            # Group by date and payment method
            payment_methods = sales_payments_df.groupby([
                pd.to_datetime(sales_payments_df["Payment Date"]).dt.date,
                "Payment Method"
            ]).agg({
                "Payment Amount": "sum",
                "Sale ID": "count"
            }).reset_index()
            
            # Rename columns
            payment_methods.columns = ["date", "payment_method", "amount", "transaction_count"]
            
            # Add company column
            payment_methods["company"] = company
            
            # Ensure date is datetime
            payment_methods["date"] = pd.to_datetime(payment_methods["date"])
            
            # Add derived time features
            payment_methods['dayofweek'] = payment_methods['date'].dt.dayofweek
            payment_methods['month'] = payment_methods['date'].dt.month
            payment_methods['year'] = payment_methods['date'].dt.year
            payment_methods['is_weekend'] = (payment_methods['dayofweek'] >= 5).astype(int)
            
            data_logger.info(f"Processed payment methods for {company}: {len(payment_methods)} rows")
            return payment_methods
            
        except Exception as e:
            data_logger.error(f"Error processing payment methods: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _save_processed_data(self, company: str, data: Dict[str, pd.DataFrame]) -> None:
        """
        Save processed data to CSV files
        
        Args:
            company: Company name
            data: Dictionary of processed DataFrames
        """
        try:
            if self.use_s3:
                # Save to S3
                for key, df in data.items():
                    s3_key = f"{settings.S3_PROCESSED_PREFIX}{company}/{key}.csv"
                    self.s3_service.write_csv_file(df, s3_key)
                    data_logger.info(f"Saved processed data to S3: {s3_key}")
            else:
                # Save to local filesystem
                company_dir = os.path.join(self.processed_dir, company)
                os.makedirs(company_dir, exist_ok=True)
                
                for key, df in data.items():
                    file_path = os.path.join(company_dir, f"{key}.csv")
                    df.to_csv(file_path, index=False)
                    data_logger.info(f"Saved processed data to: {file_path}")
                
        except Exception as e:
            data_logger.error(f"Error saving processed data: {str(e)}")
            data_logger.error(traceback.format_exc())
    
    def load_processed_data(self, company: str) -> Dict[str, pd.DataFrame]:
        """
        Load processed data for a company
        
        Args:
            company: Company name
            
        Returns:
            Dictionary of processed DataFrames
        """
        try:
            result = {}
            
            if self.use_s3:
                # Load from S3
                prefix = f"{settings.S3_PROCESSED_PREFIX}{company}/"
                files = self.s3_service.list_objects(prefix)
                
                for s3_key in files:
                    if s3_key.endswith(".csv"):
                        key = os.path.basename(s3_key).split(".")[0]
                        
                        # Special handling for product_category_map and other files without date column
                        if key == "product_category_map":
                            df = self.s3_service.read_csv_file(s3_key)
                        else:
                            # For files with date columns, attempt to parse dates safely
                            try:
                                df = self.s3_service.read_csv_file(s3_key, parse_dates=["date"])
                            except Exception as e:
                                data_logger.warning(f"Error parsing dates for {s3_key}: {str(e)}. Trying without date parsing.")
                                df = self.s3_service.read_csv_file(s3_key)
                                
                        result[key] = df
                        data_logger.info(f"Loaded processed data from S3: {s3_key}")
            else:
                # Load from local filesystem
                company_dir = os.path.join(self.processed_dir, company)
                
                if not os.path.exists(company_dir):
                    data_logger.warning(f"No processed data directory for company: {company}")
                    return {}
                
                for file_name in os.listdir(company_dir):
                    if file_name.endswith(".csv"):
                        key = file_name.split(".")[0]
                        file_path = os.path.join(company_dir, file_name)
                        
                        # Check if this is a file with date columns or not
                        # First, peek at the header to see if 'date' column exists
                        header = pd.read_csv(file_path, nrows=0).columns.tolist()
                        
                        if 'date' in header:
                            # If 'date' column exists, parse it
                            result[key] = pd.read_csv(file_path, parse_dates=["date"])
                        else:
                            # Otherwise, just load the file without date parsing
                            result[key] = pd.read_csv(file_path)
                            
                        data_logger.info(f"Loaded processed data from: {file_path}")
            
            return result
            
        except Exception as e:
            data_logger.error(f"Error loading processed data: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {}

def test_data_processor():
    """Test the DataProcessor functionality"""
    from app.data.loader import DataLoader
    
    loader = DataLoader()
    processor = DataProcessor()
    
    # Test processing data for Forge
    company = "forge"
    raw_data = loader.load_company_data(company)
    
    if raw_data:
        processed_data = processor.process_company_data(company, raw_data)
        
        # Log processed data stats
        for key, df in processed_data.items():
            print(f"{key}: {len(df)} rows, {df.columns.tolist()}")
            if not df.empty:
                print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Test loading processed data
        loaded_data = processor.load_processed_data(company)
        print(f"Loaded data types: {list(loaded_data.keys())}")
    
    return "Data processor test completed successfully"

if __name__ == "__main__":
    test_data_processor()