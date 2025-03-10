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

class DataLoader:
    """
    Service for loading sales data from Excel files for Sheffield companies
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize the data loader"""
        self.data_dir = data_dir
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
            full_path = os.path.join(self.data_dir, file_path)
            data_logger.info(f"Loading Excel file: {full_path}")
            
            if not os.path.exists(full_path):
                data_logger.error(f"File not found: {full_path}")
                return pd.DataFrame()
            
            return pd.read_excel(full_path)
        except Exception as e:
            data_logger.error(f"Error loading Excel file {file_path}: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
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