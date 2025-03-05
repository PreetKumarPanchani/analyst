import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional

def get_excel_files(directory: str, company: Optional[str] = None) -> List[str]:
    """
    Get all Excel files in the specified directory.
    Optionally filter by company name in the filename.
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith('.xlsx') and (company is None or file.startswith(company.lower())):
            files.append(os.path.join(directory, file))
    return files

def parse_excel_date(date_str: str) -> datetime:
    """Parse date strings from Excel headers."""
    # Try different date formats
    formats = [
        "%d/%m/%Y %I:%M %p",  # 01/01/2024 01:00 AM
        "%m/%d/%Y %I:%M %p",  # 01/01/2024 01:00 AM (US format)
        "%Y-%m-%d"            # 2024-01-01
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse date: {date_str}")

def read_excel_file(file_path: str) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Read an Excel file and extract metadata, sales and items data.
    Returns a tuple of (metadata, sales_df, items_df)
    """
    # Read the workbook
    workbook = pd.ExcelFile(file_path)
    
    # Get the 'Sales' sheet for metadata and sales data
    sales_df = pd.read_excel(workbook, sheet_name='Sales', header=None)
    
    # Extract metadata from first rows
    metadata = {
        'client': sales_df.iloc[1, 1] if pd.notna(sales_df.iloc[1, 1]) else None,
        'from_date': parse_excel_date(sales_df.iloc[2, 2]) if pd.notna(sales_df.iloc[2, 2]) else None,
        'to_date': parse_excel_date(sales_df.iloc[3, 2]) if pd.notna(sales_df.iloc[3, 2]) else None,
    }
    
    # Extract actual sales data (skipping header rows)
    sales_headers = sales_df.iloc[4]
    sales_df = pd.read_excel(workbook, sheet_name='Sales', header=4)
    
    # Extract items data
    items_df = pd.read_excel(workbook, sheet_name='Sales Items', header=4)
    
    return metadata, sales_df, items_df

def clean_sales_data(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare sales DataFrame for analysis.
    """
    # Make a copy to avoid modifying the original
    df = sales_df.copy()
    
    # Convert dates if they're not already datetime objects
    if df['Sale Date'].dtype != 'datetime64[ns]':
        df['Sale Date'] = pd.to_datetime(df['Sale Date'], errors='coerce')
    
    # Drop rows with missing key values
    df = df.dropna(subset=['Sale ID', 'Sale Date', 'Total'])
    
    # Convert numeric columns
    numeric_cols = ['Quantity', 'Net', 'VAT', 'Total']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def clean_items_data(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare items DataFrame for analysis.
    """
    # Make a copy to avoid modifying the original
    df = items_df.copy()
    
    # Convert dates if they're not already datetime objects
    if 'Sale Date' in df.columns and df['Sale Date'].dtype != 'datetime64[ns]':
        df['Sale Date'] = pd.to_datetime(df['Sale Date'], errors='coerce')
    
    # Drop rows with missing key values
    df = df.dropna(subset=['Sale ID', 'Product Name'])
    
    # Convert numeric columns
    numeric_cols = ['Quantity', 'Unit Price', 'Total']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def merge_sales_items(sales_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sales and items data on Sale ID.
    """
    return pd.merge(sales_df, items_df, on='Sale ID', how='inner', suffixes=('', '_item'))

# Test function
def test_excel_utils():
    """Simple test function for the excel utilities."""
    test_directory = "./test_data"
    
    # Create test directory if it doesn't exist
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
        print(f"Created test directory: {test_directory}")
        print("Please place test Excel files in this directory and run the test again.")
        return
    
    # Get Excel files
    files = get_excel_files(test_directory)
    if not files:
        print(f"No Excel files found in {test_directory}")
        return
    
    # Test reading the first file
    file_path = files[0]
    print(f"Testing file: {file_path}")
    
    try:
        metadata, sales_df, items_df = read_excel_file(file_path)
        print("Metadata:", metadata)
        print("\nSales Data (first 2 rows):")
        print(sales_df.head(2))
        print("\nItems Data (first 2 rows):")
        print(items_df.head(2))
        
        # Test cleaning
        clean_sales = clean_sales_data(sales_df)
        clean_items = clean_items_data(items_df)
        
        print("\nCleaned Sales Data (first 2 rows):")
        print(clean_sales.head(2))
        print("\nCleaned Items Data (first 2 rows):")
        print(clean_items.head(2))
        
        print("\nTest successful!")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    test_excel_utils()