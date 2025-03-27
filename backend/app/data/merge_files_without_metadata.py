# merge_files_without_metadata.py
import pandas as pd
import os
import re
import numpy as np

def get_data_without_metadata(file_path, sheet_name, company_name= None):
    """
    Extract only the data rows from an Excel file, ignoring metadata
    Returns a pandas dataframe with the actual data
    """
    try:
        # Read the entire sheet into a dataframe
        full_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # Find the row that contains "Sale ID" or similar header
        header_row = None
        header_indicators = ["Sale ID", "SaleID", "Sale_ID", "Sale Date"]
        
        for i, row in full_df.iterrows():
            # Convert row to string and check if any cell contains a header indicator
            row_as_str = row.astype(str).str.lower()
            if any(indicator.lower() in val for indicator in header_indicators for val in row_as_str):
                header_row = i
                break
        

        ## If header row is not found then take the first row where multiple values/columns are present , not just one column or value 
        if header_row is None:
            # Look for the first row with multiple non-empty values
            for i, row in full_df.iterrows():
                # Count non-empty cells in this row
                non_empty_cells = row.notna().sum()
                
                # If we have more than 2 non-empty cells, consider this a header row
                if non_empty_cells >= 3:
                    header_row = i
                    print(f"  Found potential header row at position {i+1} with {non_empty_cells} values in {file_path}, sheet {sheet_name}.")
                    break



        if header_row is None:
            # If we couldn't find the header row, assume it's row 5 (common in these files)
            header_row = 5
            print(f"  Warning: Couldn't find header row in {file_path}, sheet {sheet_name}. Assuming row 6.")
        
        # Extract the actual data (including header row)
        data_df = full_df.iloc[header_row:].copy().reset_index(drop=True)
        
        # Set the first row as header
        data_df.columns = data_df.iloc[0]
        data_df = data_df.iloc[1:].reset_index(drop=True)
        
        # 
        # Add a column to track the source file
        data_df['Source_File'] = os.path.basename(file_path)
        
        if company_name is not None:
            data_df['Company'] = company_name.upper()
        else:
            # Extract company from filename
            company_match = re.search(r'(cpl|forge)_', os.path.basename(file_path).lower())
            if company_match:
                data_df['Company'] = company_match.group(1).upper()
            else:
                data_df['Company'] = "UNKNOWN"
            
        return data_df
    
    except Exception as e:
        print(f"Error processing {file_path}, sheet {sheet_name}: {e}")
        return pd.DataFrame()

## Sheet names should be same across all files and column names should be same across all files 

def process_files(files, company, sheet_names, output_dir):
    """Process files for a specific company and generate merged outputs without metadata"""
    # Dictionary to hold dataframes for each sheet
    all_data = {sheet: [] for sheet in sheet_names}
    
    # Process each file
    print(f"\nProcessing {len(files)} {company.upper()} files")
    for file in files:
        try:
            print(f"Reading file: {file}")
            for sheet in sheet_names:
                try:
                    if company is not None: 
                        data_df = get_data_without_metadata(file, sheet, company)
                    else:
                        data_df = get_data_without_metadata(file, sheet)
                    
                    if not data_df.empty:
                        # Convert all column names to string to avoid issues
                        data_df.columns = data_df.columns.astype(str)
                        
                        # Store data
                        all_data[sheet].append(data_df)
                        print(f"  Extracted {len(data_df)} rows from {sheet}")
                    else:
                        print(f"  No data found in sheet {sheet}")
                        
                except Exception as e:
                    print(f"  Error processing sheet {sheet} in {file}: {e}")
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    # Create merged files for each sheet
    for sheet in sheet_names:
        if not all_data[sheet]:
            print(f"No data found for {sheet} in any {company} file")
            continue
            
        try:
            # Merge all data for this sheet
            # Find all unique columns across all dataframes
            all_columns = set()
            for df in all_data[sheet]:
                all_columns.update(df.columns)
            
            # Ensure all dataframes have all columns
            for i, df in enumerate(all_data[sheet]):
                missing_cols = all_columns - set(df.columns)
                if missing_cols:
                    for col in missing_cols:
                        df[col] = np.nan
            
            # Concatenate all data
            merged_data = pd.concat(all_data[sheet], ignore_index=True)
            
            # Create output Excel file
            output_file = os.path.join(output_dir, f"{company}_{sheet.replace(' ', '_')}.xlsx")
            
            # Save the dataframe directly to Excel without metadata
            merged_data.to_excel(output_file, index=False)
            print(f"Created {output_file} with {len(merged_data)} rows")
            
        except Exception as e:
            print(f"Error creating merged file for {sheet}: {e}")
    
    return all_data

def verify_merge(all_data, company):
    """Verify that data from all files was merged correctly"""
    print(f"\nVerification for {company.upper()}:")
    
    for sheet, data_list in all_data.items():
        if not data_list:
            print(f"No data found for {sheet}")
            continue
        
        total_rows = sum(len(df) for df in data_list)
        unique_files = set()
        
        for df in data_list:
            unique_files.update(df['Source_File'].unique())
        
        print(f"{sheet}: Merged data from {len(unique_files)} files, total rows: {total_rows}")
        print(f"  Source files: {', '.join(sorted(unique_files))}")




# Main function
def main():
    from app.core.config import Settings
    settings = Settings()
    dir_path = settings.RAW_DATA_DIR

    # Define files
    cpl_files = ["cpl_1-2_export_transactions_report_20250222_195204.xlsx",
                 "cpl_3-5_export_transactions_report_20250222_195244.xlsx",
                 "cpl_6-8_export_transactions_report_20250222_195343.xlsx", 
                 "cpl_9-11_export_transactions_report_20250222_195434.xlsx",
                 "cpl_12-14_export_transactions_report_20250222_195503.xlsx",
                 ]

    forge_files = ["forge_1-2_export_transactions_report_20250222_195117.xlsx",
                   "forge_3-4_export_transactions_report_20250222_194908.xlsx",
                   "forge_5-6_export_transactions_report_20250222_194829.xlsx",
                   "forge_7-8_export_transactions_report_20250222_194803.xlsx",
                   "forge_9-10_export_transactions_report_20250222_194731.xlsx",
                   "forge_11-12_export_transactions_report_20250222_194658.xlsx",
                   "forge_13-14_export_transactions_report_20250222_194607.xlsx"]

    # Define sheet names
    sheet_names = ['Sales', 'Sales Items', 'Sales Payments', 'Sales Refunds', 'Deleted Sales Items']
    
    # Add the output directory to the file paths
    cpl_files = [os.path.join(dir_path, f) for f in cpl_files]
    forge_files = [os.path.join(dir_path, f) for f in forge_files]

    # Create output directory
    output_dir = settings.MERGED_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Process CPL files
    cpl_data = process_files(cpl_files, "cpl", sheet_names, output_dir)
    
    # Process Forge files
    forge_data = process_files(forge_files, "forge", sheet_names, output_dir)
    
    # Verify the merges
    verify_merge(cpl_data, "cpl")
    verify_merge(forge_data, "forge")
    
    print("\nMerging completed successfully.")
    print(f"The merged files can be found in the '{output_dir}' directory.")



if __name__ == "__main__":
    main()