# app/data/loader.py
import os
import pandas as pd
from typing import Dict, List, Optional
import traceback

import sys
#sys.path.append(os.path.dirname((os.path.abspath(__file__))))
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.logger import data_logger
from app.core.config import settings
from app.services.s3_service import S3Service

class DataLoader:
    """
    Service for loading sales data from Excel files for Sheffield companies
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the data loader"""
        self.data_dir = data_dir
        self.s3_service = S3Service()
        self.use_s3 = settings.USE_S3_STORAGE
        data_logger.info(f"DataLoader initialized with data directory: {data_dir}")
    
    def load_excel_file(self, file_path: str) -> pd.DataFrame:
        """
        Load an Excel file into a DataFrame
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame with the Excel data
        """
        try:
            if self.use_s3:
                # Use S3 path
                s3_key = f"{settings.S3_RAW_PREFIX}{file_path}"
                data_logger.info(f"Loading Excel file from S3: {s3_key}")
                return self.s3_service.read_excel_file(s3_key)
            else:
                # Use local filesystem
                full_path = os.path.join(self.data_dir, file_path)
                data_logger.info(f"Loading Excel file from local filesystem: {full_path}")
                
                if not os.path.exists(full_path):
                    data_logger.error(f"File not found: {full_path}")
                    return pd.DataFrame()
                
                return pd.read_excel(full_path)
        except Exception as e:
            data_logger.error(f"Error loading Excel file {file_path}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    '''
    def load_company_data(self, company: str) -> Dict[str, pd.DataFrame]:
        """
        Load all data files for a specific company
        
        Args:
            company: Company name (forge or cpl)
            
        Returns:
            Dictionary of DataFrames for different data types
        """
        try:
            data_logger.info(f"Loading all data for company: {company}")
            
            file_prefix = company.lower()
            result = {}
            
            # Load Sales data
            sales_file = f"{file_prefix}_Sales.xlsx"
            result["sales"] = self.load_excel_file(sales_file)
            
            # Load Sales Items data
            sales_items_file = f"{file_prefix}_Sales_Items.xlsx"
            result["sales_items"] = self.load_excel_file(sales_items_file)
            
            # Load Sales Payments data
            sales_payments_file = f"{file_prefix}_Sales_Payments.xlsx"
            result["sales_payments"] = self.load_excel_file(sales_payments_file)
            
            # Load Deleted Sales Items data
            deleted_sales_items_file = f"{file_prefix}_Deleted_Sales_Items.xlsx"
            result["deleted_sales_items"] = self.load_excel_file(deleted_sales_items_file)
            
            data_logger.info(f"Successfully loaded data for company: {company}")
            return result
            
        except Exception as e:
            data_logger.error(f"Error loading company data for {company}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {}
    '''



    def load_company_data(self, company: str) -> Dict[str, pd.DataFrame]:
        """
        Load all data files for a specific company, returning empty DataFrames if files don't exist
        
        Args:
            company: Company name (forge or cpl)
            
        Returns:
            Dictionary of DataFrames for different data types (empty if no data)
        """
        try:
            data_logger.info(f"Safely loading all data for company: {company}")
            
            file_prefix = company.lower()
            result = {
                "sales": pd.DataFrame(),
                "sales_items": pd.DataFrame(),
                "sales_payments": pd.DataFrame(),
                "deleted_sales_items": pd.DataFrame()
            }
            
            # Try to load each file, but don't fail if missing
            try:
                sales_file = f"{file_prefix}_Sales.xlsx"
                df = self.load_excel_file(sales_file)
                if not df.empty:
                    result["sales"] = df
            except Exception as e:
                data_logger.warning(f"Could not load sales data for {company}: {str(e)}")
            
            try:
                sales_items_file = f"{file_prefix}_Sales_Items.xlsx"
                df = self.load_excel_file(sales_items_file)
                if not df.empty:
                    result["sales_items"] = df
            except Exception as e:
                data_logger.warning(f"Could not load sales items data for {company}: {str(e)}")
            
            try:
                sales_payments_file = f"{file_prefix}_Sales_Payments.xlsx"
                df = self.load_excel_file(sales_payments_file)
                if not df.empty:
                    result["sales_payments"] = df
            except Exception as e:
                data_logger.warning(f"Could not load sales payments data for {company}: {str(e)}")
            
            try:
                deleted_sales_items_file = f"{file_prefix}_Deleted_Sales_Items.xlsx"
                df = self.load_excel_file(deleted_sales_items_file)
                if not df.empty:
                    result["deleted_sales_items"] = df
            except Exception as e:
                data_logger.warning(f"Could not load deleted sales items data for {company}: {str(e)}")
                
            # Check if we have any data
            has_data = any(not df.empty for df in result.values())
            data_logger.info(f"Successfully loaded data for company: {company}, has data: {has_data}")
            
            return result
                
        except Exception as e:
            data_logger.error(f"Error loading company data for {company}: {str(e)}")
            data_logger.error(traceback.format_exc())
            
            # Return empty DataFrames
            return {
                "sales": pd.DataFrame(),
                "sales_items": pd.DataFrame(),
                "sales_payments": pd.DataFrame(),
                "deleted_sales_items": pd.DataFrame()
            }
    
        

    def has_company_data(self, company: str) -> bool:
        """
        Check if data exists for a specific company
        
        Args:
            company: Company name (forge or cpl)
            
        Returns:
            True if data exists, False otherwise
        """
        try:
            data_logger.info(f"Checking if data exists for company: {company}")
            has_data = False
            if self.use_s3:
                # Check S3 for raw data files
                file_prefix = f"{settings.S3_RAW_PREFIX}{company.lower()}_"
                files = self.s3_service.list_objects(file_prefix)
                
                # Data exists if we found at least one file
                has_data = len(files) > 0
    

                #response = s3_service.s3_client.list_objects_v2(
                #    Bucket=s3_service.bucket_name,
                #    Prefix=prefix
                #)
                
                #has_data = 'Contents' in response and len(response['Contents']) > 0

            else:
                # Check local filesystem for raw data files
                file_patterns = [
                    f"{company.lower()}_Sales.xlsx",
                    f"{company.lower()}_Sales_Items.xlsx",
                    f"{company.lower()}_Sales_Payments.xlsx",
                    f"{company.lower()}_Deleted_Sales_Items.xlsx"
                ]
                
                has_data = False
                i = 0
                for pattern in file_patterns:
                    file_path = os.path.join(settings.RAW_DATA_DIR, pattern)
                    if os.path.exists(file_path):
                        i += 1
                if i > 0:
                    has_data = True


            
            data_logger.info(f"Data exists for company {company}: {has_data}")
            return has_data
            
        except Exception as e:
            data_logger.error(f"Error checking if data exists for {company}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
        

def test_data_loader():
    """Test the DataLoader functionality"""
    loader = DataLoader()
    
    # Test loading data for Forge
    forge_data = loader.load_company_data("forge")
    
    # Log data size for each file
    for key, df in forge_data.items():
        print(f"{key}: {len(df)} rows, {len(df.columns)} columns")
        print(f"columns: {', '.join(list(df.columns)[:5])}")
    
    # Test loading data for CPL
    cpl_data = loader.load_company_data("cpl")
    
    # Log data size for each file
    for key, df in cpl_data.items():
        print(f"{key}: {len(df)} rows, {len(df.columns)} columns")
    
    return "Data loader test completed successfully"

if __name__ == "__main__":
    test_data_loader()