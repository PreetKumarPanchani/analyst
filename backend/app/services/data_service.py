import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from app.utils.excel_utils import get_excel_files, read_excel_file, clean_sales_data, clean_items_data

class DataService:
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the data service with the directory containing Excel files.
        """
        self.data_dir = data_dir
        self.forge_data = {"sales": pd.DataFrame(), "items": pd.DataFrame()}
        self.cpl_data = {"sales": pd.DataFrame(), "items": pd.DataFrame()}
        self.loaded = False
    
    def load_data(self, force_reload: bool = False) -> bool:
        """
        Load all data from Excel files in the data directory.
        Returns True if data was loaded successfully.
        """
        if self.loaded and not force_reload:
            return True
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")
            return False
        
        # Get all Excel files
        forge_files = get_excel_files(self.data_dir, company="forge")
        cpl_files = get_excel_files(self.data_dir, company="cpl")
        
        if not forge_files and not cpl_files:
            print(f"No Excel files found in {self.data_dir}")
            return False
        
        # Process Forge files
        forge_sales_dfs = []
        forge_items_dfs = []
        
        for file in forge_files:
            try:
                _, sales_df, items_df = read_excel_file(file)
                forge_sales_dfs.append(clean_sales_data(sales_df))
                forge_items_dfs.append(clean_items_data(items_df))
            except Exception as e:
                print(f"Error processing Forge file {file}: {e}")
        
        # Process CPL files
        cpl_sales_dfs = []
        cpl_items_dfs = []
        
        for file in cpl_files:
            try:
                _, sales_df, items_df = read_excel_file(file)
                cpl_sales_dfs.append(clean_sales_data(sales_df))
                cpl_items_dfs.append(clean_items_data(items_df))
            except Exception as e:
                print(f"Error processing CPL file {file}: {e}")
        
        # Combine data frames
        if forge_sales_dfs:
            self.forge_data["sales"] = pd.concat(forge_sales_dfs, ignore_index=True)
        if forge_items_dfs:
            self.forge_data["items"] = pd.concat(forge_items_dfs, ignore_index=True)
        
        if cpl_sales_dfs:
            self.cpl_data["sales"] = pd.concat(cpl_sales_dfs, ignore_index=True)
        if cpl_items_dfs:
            self.cpl_data["items"] = pd.concat(cpl_items_dfs, ignore_index=True)
        
        # Mark as loaded
        self.loaded = True
        return True
    
    def get_company_data(self, company: str) -> Dict[str, pd.DataFrame]:
        """
        Get the data for a specific company.
        """
        if not self.loaded:
            self.load_data()
        
        if company.lower() == "forge":
            return self.forge_data
        elif company.lower() == "cpl":
            return self.cpl_data
        else:
            raise ValueError(f"Invalid company: {company}. Expected 'forge' or 'cpl'.")
    
    def get_sales_summary(self, company: str) -> Dict[str, Any]:
        """
        Get summary statistics for sales.
        """
        data = self.get_company_data(company)
        sales_df = data["sales"]
        
        if sales_df.empty:
            return {
                "total_sales": 0,
                "transaction_count": 0,
                "avg_basket": 0,
                "start_date": None,
                "end_date": None
            }
        
        return {
            "total_sales": float(sales_df["Total"].sum()),
            "transaction_count": len(sales_df),
            "avg_basket": float(sales_df["Total"].mean()),
            "start_date": sales_df["Sale Date"].min(),
            "end_date": sales_df["Sale Date"].max()
        }
    
    def get_monthly_sales(self, company: str) -> List[Dict[str, Any]]:
        """
        Get monthly sales statistics.
        """
        data = self.get_company_data(company)
        sales_df = data["sales"]
        
        if sales_df.empty:
            return []
        
        # Add year-month column for grouping
        sales_df['YearMonth'] = sales_df['Sale Date'].dt.strftime('%Y-%m')
        
        # Group by year-month
        monthly = sales_df.groupby('YearMonth').agg(
            sales=('Total', 'sum'),
            transactions=('Sale ID', 'count')
        ).reset_index()
        
        # Calculate average basket
        monthly['avg_basket'] = monthly['sales'] / monthly['transactions']
        
        # Sort by year-month
        monthly = monthly.sort_values('YearMonth')
        
        # Convert to list of dictionaries
        return monthly.to_dict('records')
    
    def get_daily_sales(self, company: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily sales statistics for the last N days.
        """
        data = self.get_company_data(company)
        sales_df = data["sales"]
        
        if sales_df.empty:
            return []
        
        # Calculate cutoff date
        end_date = sales_df["Sale Date"].max()
        start_date = end_date - timedelta(days=days)
        
        # Filter for last N days
        recent_sales = sales_df[sales_df["Sale Date"] >= start_date]
        
        # Group by date
        daily = recent_sales.groupby(recent_sales['Sale Date'].dt.date).agg(
            sales=('Total', 'sum'),
            transactions=('Sale ID', 'count')
        ).reset_index()
        
        # Calculate average basket
        daily['avg_basket'] = daily['sales'] / daily['transactions']
        
        # Sort by date
        daily = daily.sort_values('Sale Date')
        
        # Convert to list of dictionaries
        return daily.to_dict('records')
    
    def get_register_summary(self, company: str) -> List[Dict[str, Any]]:
        """
        Get sales summary by register.
        """
        data = self.get_company_data(company)
        sales_df = data["sales"]
        
        if sales_df.empty:
            return []
        
        # Group by register
        registers = sales_df.groupby('Register').agg(
            sales=('Total', 'sum'),
            transactions=('Sale ID', 'count')
        ).reset_index()
        
        # Calculate average basket
        registers['avg_basket'] = registers['sales'] / registers['transactions']
        
        # Sort by sales in descending order
        registers = registers.sort_values('sales', ascending=False)
        
        # Convert to list of dictionaries
        return registers.to_dict('records')
    
    def get_top_products(self, company: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top N products by sales.
        """
        data = self.get_company_data(company)
        items_df = data["items"]
        
        if items_df.empty:
            return []
        
        # Group by product name
        products = items_df.groupby(['Product Name', 'Category']).agg(
            sales=('Total', 'sum'),
            quantity=('Quantity', 'sum')
        ).reset_index()
        
        # Sort by sales in descending order
        products = products.sort_values('sales', ascending=False)
        
        # Limit to top N
        products = products.head(limit)
        
        # Convert to list of dictionaries
        return products.to_dict('records')
    
    def get_category_summary(self, company: str) -> List[Dict[str, Any]]:
        """
        Get sales summary by category.
        """
        data = self.get_company_data(company)
        items_df = data["items"]
        
        if items_df.empty:
            return []
        
        # Group by category
        categories = items_df.groupby('Category').agg(
            sales=('Total', 'sum'),
            quantity=('Quantity', 'sum'),
            product_count=('Product Name', 'nunique')
        ).reset_index()
        
        # Sort by sales in descending order
        categories = categories.sort_values('sales', ascending=False)
        
        # Convert to list of dictionaries
        return categories.to_dict('records')
    
    def get_daily_data_for_forecast(self, company: str) -> pd.DataFrame:
        """
        Prepare data for time series forecasting.
        Returns DataFrame with 'ds' and 'y' columns for Prophet.
        """
        data = self.get_company_data(company)
        sales_df = data["sales"]
        
        if sales_df.empty:
            return pd.DataFrame(columns=['ds', 'y'])
        
        # Group by date
        daily = sales_df.groupby(sales_df['Sale Date'].dt.date).agg(
            y=('Total', 'sum')
        ).reset_index()
        
        # Rename the date column to 'ds' for Prophet
        daily = daily.rename(columns={'Sale Date': 'ds'})
        
        return daily


# Test function
def test_data_service():
    """Simple test function for the data service."""
    # Initialize the data service with test data directory
    service = DataService("./test_data")
    
    # Test loading data
    loaded = service.load_data()
    if not loaded:
        print("Data loading failed or no data available")
        return
    
    # Test getting company data
    try:
        forge_data = service.get_company_data("forge")
        print(f"Forge data loaded: {len(forge_data['sales'])} sales records, {len(forge_data['items'])} item records")
    except Exception as e:
        print(f"Error getting Forge data: {e}")
    
    try:
        cpl_data = service.get_company_data("cpl")
        print(f"CPL data loaded: {len(cpl_data['sales'])} sales records, {len(cpl_data['items'])} item records")
    except Exception as e:
        print(f"Error getting CPL data: {e}")
    
    # Test getting sales summary
    try:
        forge_summary = service.get_sales_summary("forge")
        print("\nForge Sales Summary:")
        print(f"Total Sales: £{forge_summary['total_sales']:.2f}")
        print(f"Transaction Count: {forge_summary['transaction_count']}")
        print(f"Average Basket: £{forge_summary['avg_basket']:.2f}")
        print(f"Date Range: {forge_summary['start_date']} to {forge_summary['end_date']}")
    except Exception as e:
        print(f"Error getting Forge sales summary: {e}")
    
    # Test getting top products
    try:
        top_products = service.get_top_products("forge", limit=5)
        print("\nForge Top 5 Products:")
        for i, product in enumerate(top_products, 1):
            print(f"{i}. {product['Product Name']} ({product['Category']}): £{product['sales']:.2f}, {product['quantity']} units")
    except Exception as e:
        print(f"Error getting Forge top products: {e}")
    
    print("\nTest completed successfully")

# Run test if this file is executed directly
if __name__ == "__main__":
    test_data_service()