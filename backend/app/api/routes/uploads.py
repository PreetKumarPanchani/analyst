# app/api/routes/uploads.py
import os
import shutil
import uuid
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from typing import List, Dict, Any, Optional
import traceback
import sys
from pathlib import Path

# Import the process_files function from merge_files_without_metadata
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.data.merge_files_without_metadata import process_files, get_data_without_metadata

from app.core.logger import api_logger
from app.core.config import settings
from app.services.s3_service import S3Service
from app.data.loader import DataLoader
from app.data.processor import DataProcessor

router = APIRouter()
s3_service = S3Service()
data_loader = DataLoader()
data_processor = DataProcessor()

@router.post("/file")
async def upload_file(
    company: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload an Excel file for processing
    
    Args:
        company: Company name (forge or cpl)
        file: Excel file to upload
    """
    try:
        api_logger.info(f"Received file upload for company: {company}")
        
        # Validate company
        if company not in ["forge", "cpl"]:
            raise HTTPException(status_code=400, detail=f"Invalid company: {company}. Must be 'forge' or 'cpl'")
        
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
        
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{company}_{timestamp}_{unique_id}.xlsx"
        
        # Save file to new_data directory (create if doesn't exist)
        if settings.USE_S3_STORAGE:
            # Save to S3
            file_content = await file.read()
            s3_key = f"new_data/{filename}"
            
            s3_service.s3_client.put_object(
                Bucket=s3_service.bucket_name,
                Key=s3_key,
                Body=file_content
            )
            
            file_path = s3_key
            api_logger.info(f"File saved to S3: {s3_key}")

            # Setup output directory for processed files
            # Use S3 prefix for saving to S3 later, but ensure local directory exists for temporary processing
            output_dir_s3 = settings.S3_MERGED_DATA_PREFIX
            output_dir_local = os.path.join(settings.DATA_DIR, "merged_dataset_without_metadata")
            os.makedirs(output_dir_local, exist_ok=True)
            api_logger.info(f"Created local merged directory: {output_dir_local}")
        else:
            # Save to local filesystem
            new_data_dir = os.path.join(settings.BASE_DIR, "data", "new_data")
            os.makedirs(new_data_dir, exist_ok=True)
            
            file_path = os.path.join(new_data_dir, filename)
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
                
            api_logger.info(f"File saved locally: {file_path}")
            
            # Process the file using functions from merge_files_without_metadata
            output_dir_local = settings.MERGED_DATA_DIR
            os.makedirs(output_dir_local, exist_ok=True)
            
        api_logger.info(f"Processing file: {filename}")
        
        # Process the file - for now, just call with a single file
        if settings.USE_S3_STORAGE:
            # Download the file from S3 to process
            temp_file_path = os.path.join(settings.BASE_DIR, "data", "temp", filename)
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            
            s3_service.s3_client.download_file(
                Bucket=s3_service.bucket_name,
                Key=s3_key,
                Filename=temp_file_path
            )
            
            # Process the file - use local directory for temporary file creation
            files = [temp_file_path]
            all_data = process_files(files, company, ["Sales", "Sales Items", "Sales Payments", "Sales Refunds", "Deleted Sales Items"], output_dir_local)
            
            # Upload processed files from local to S3
            api_logger.info(f"Uploading processed files from {output_dir_local} to S3 prefix {output_dir_s3}")
            for sheet_type in ["Sales", "Sales Items", "Sales Payments", "Sales Refunds", "Deleted Sales Items"]:
                sheet_file = sheet_type.replace(' ', '_')
                local_file_path = os.path.join(output_dir_local, f"{company}_{sheet_file}.xlsx")
                
                if os.path.exists(local_file_path):
                    s3_key = f"{output_dir_s3}{company}_{sheet_file}.xlsx"
                    s3_service.s3_client.upload_file(
                        Filename=local_file_path,
                        Bucket=s3_service.bucket_name,
                        Key=s3_key
                    )
                    api_logger.info(f"Uploaded processed file to S3: {s3_key}")
            
            # Clean up temp file
            os.remove(temp_file_path)
        else:
            # Process local file
            files = [file_path]
            all_data = process_files(files, company, ["Sales", "Sales Items", "Sales Payments", "Sales Refunds", "Deleted Sales Items"], output_dir_local)

        # Merge with existing data
        await merge_with_existing_data(company, output_dir_local)
        
        # Clean up processed data, cache, and models for this company
        await cleanup_company_data(company)
        
        return {
            "success": True,
            "message": f"File processed successfully for {company}",
            "file_path": file_path
        }
    except Exception as e:
        api_logger.error(f"Error processing uploaded file: {str(e)}")
        api_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

async def merge_with_existing_data(company: str, output_dir: str):
    """Merge new data with existing data based on Sale Date and Sale Time"""
    try:
        api_logger.info(f"Merging new data with existing for company: {company}")
        
        # Get paths for sheet types
        sheet_types = ["Sales", "Sales Items", "Sales Payments", "Deleted Sales Items"]
        
        for sheet in sheet_types:
            # Clean sheet name for filename
            sheet_file = sheet.replace(' ', '_')
            
            # New data file path
            new_data_path = os.path.join(output_dir, f"{company}_{sheet_file}.xlsx")
            
            # Skip if the file doesn't exist
            if not os.path.exists(new_data_path):
                api_logger.warning(f"New data file not found: {new_data_path}")
                continue
                
            # Existing data file path
            if settings.USE_S3_STORAGE:
                # For S3 storage
                existing_file_key = f"{settings.S3_RAW_PREFIX}{company}_{sheet_file}.xlsx"
                
                # Check if the file exists in S3
                try:
                    s3_service.s3_client.head_object(
                        Bucket=s3_service.bucket_name,
                        Key=existing_file_key
                    )
                    file_exists = True
                except:
                    file_exists = False
                    
                if file_exists:
                    # Download the existing file
                    temp_existing_path = os.path.join(settings.BASE_DIR, "data", "temp", f"existing_{company}_{sheet_file}.xlsx")
                    os.makedirs(os.path.dirname(temp_existing_path), exist_ok=True)
                    
                    s3_service.s3_client.download_file(
                        Bucket=s3_service.bucket_name,
                        Key=existing_file_key,
                        Filename=temp_existing_path
                    )
                    
                    # Read data
                    existing_df = pd.read_excel(temp_existing_path)
                    new_df = pd.read_excel(new_data_path)
                    
                    # Merge data
                    merged_df = merge_data_by_date(existing_df, new_df)
                    
                    # Save merged data
                    merged_df.to_excel(temp_existing_path, index=False)
                    
                    # Upload back to S3
                    s3_service.s3_client.upload_file(
                        Filename=temp_existing_path,
                        Bucket=s3_service.bucket_name,
                        Key=existing_file_key
                    )
                    
                    # Clean up temp file
                    os.remove(temp_existing_path)
                else:
                    api_logger.info(f"No existing file found: {existing_file_key}, uploading new data as initial dataset")

                    # If no existing file, upload the new one to create the initial dataset
                    s3_service.s3_client.upload_file(
                        Filename=new_data_path,
                        Bucket=s3_service.bucket_name,
                        Key=existing_file_key
                    )
            else:
                # For local storage
                existing_data_path = os.path.join(settings.RAW_DATA_DIR, f"{company}_{sheet_file}.xlsx")
                
                if os.path.exists(existing_data_path):
                    # Read data
                    existing_df = pd.read_excel(existing_data_path)
                    new_df = pd.read_excel(new_data_path)
                    
                    # Merge data
                    api_logger.info(f"Merging data for {company} {sheet}")
                    merged_df = merge_data_by_date(existing_df, new_df)
                    
                    # Save merged data
                    api_logger.info(f"Saving merged data for {company} {sheet}")
                    merged_df.to_excel(existing_data_path, index=False)
                else:
                    api_logger.info(f"No existing file found: {existing_data_path}, copying new data as initial dataset")
                    # If no existing file, just copy the new one
                    shutil.copy(new_data_path, existing_data_path)
        
        api_logger.info(f"Successfully merged data for company: {company}")
        return True
    except Exception as e:
        api_logger.error(f"Error merging data: {str(e)}")
        api_logger.error(traceback.format_exc())
        return False

def merge_data_by_date(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge existing and new dataframes, keeping only new records based on Sale Date and Sale Time
    """
    if existing_df.empty:
        return new_df
        
    if new_df.empty:
        return existing_df
    
    # Check if both DataFrames have the necessary columns
    date_col = 'Sale Date'
    time_col = 'Sale Time'
    
    # If date column is not present in either DataFrame, just concatenate them
    if date_col not in existing_df.columns or date_col not in new_df.columns:
        return pd.concat([existing_df, new_df], ignore_index=True)
    
    # Create a copy of the dataframes to avoid modifying the originals
    existing_df_copy = existing_df.copy()
    new_df_copy = new_df.copy()
    
    # Ensure Sale Date contains only date information (no time)
    existing_df_copy[date_col] = pd.to_datetime(existing_df_copy[date_col], errors='coerce').dt.date
    new_df_copy[date_col] = pd.to_datetime(new_df_copy[date_col], errors='coerce').dt.date
    
    # Create datetime column if time column exists
    if time_col in existing_df_copy.columns and time_col in new_df_copy.columns:
        # Convert time to string to ensure correct handling
        existing_df_copy[time_col] = existing_df_copy[time_col].astype(str)
        new_df_copy[time_col] = new_df_copy[time_col].astype(str)
        
        # Create a combined datetime column for sorting and comparison
        existing_df_copy['combined_datetime'] = existing_df_copy.apply(
            lambda row: pd.to_datetime(f"{row[date_col]} {row[time_col]}", errors='coerce'),
            axis=1
        )
        new_df_copy['combined_datetime'] = new_df_copy.apply(
            lambda row: pd.to_datetime(f"{row[date_col]} {row[time_col]}", errors='coerce'),
            axis=1
        )
        
        # Find the latest datetime in existing data
        latest_datetime = existing_df_copy['combined_datetime'].max()
        
        # Filter new data to only include records after the latest datetime
        new_records = new_df_copy[new_df_copy['combined_datetime'] > latest_datetime]
        
        # Drop records from new data that are older than or equal to the latest in existing data
        api_logger.info(f"Filtered out/ Dropped {len(new_df_copy) - len(new_records)} records from the New Data, that were not newer than existing data")
        
        # Combine existing data with new records (using original dataframes to preserve data)
        # Get indices of the filtered records in the original new_df
        if len(new_records) > 0:
            new_indices = new_records.index
            filtered_new_df = new_df.loc[new_indices]
            result_df = pd.concat([existing_df, filtered_new_df], ignore_index=True)
            
            # Sort the final dataframe by the combined datetime for consistency
            # Create the combined datetime again for sorting
            result_df['tmp_date'] = pd.to_datetime(result_df[date_col], errors='coerce').dt.date
            result_df['tmp_time'] = result_df[time_col].astype(str)
            result_df['sort_datetime'] = result_df.apply(
                lambda row: pd.to_datetime(f"{row['tmp_date']} {row['tmp_time']}", errors='coerce'),
                axis=1
            )
            api_logger.info(f"Sorting data ...")
            result_df = result_df.sort_values('sort_datetime').reset_index(drop=True)
            

            # Drop temporary columns
            api_logger.info(f"Dropping temporary columns ...")
            result_df = result_df.drop(['tmp_date', 'tmp_time', 'sort_datetime'], axis=1)
        else:
            # If no new records to add, just return existing data
            result_df = existing_df.copy()
    else:
        api_logger.info(f"No new records to add, just returning existing data ...")
        # If time column doesn't exist, just use the date
        # Find the latest date in existing data
        latest_date = pd.to_datetime(existing_df_copy[date_col], errors='coerce').max()
        
        # Filter new data to only include records after the latest date
        new_records = new_df_copy[pd.to_datetime(new_df_copy[date_col], errors='coerce') > latest_date]
        
        # Get indices of the filtered records in the original new_df
        if len(new_records) > 0:
            new_indices = new_records.index
            filtered_new_df = new_df.loc[new_indices]
            
            # Combine existing data with new records
            result_df = pd.concat([existing_df, filtered_new_df], ignore_index=True)
            
            # Sort by date
            result_df['sort_date'] = pd.to_datetime(result_df[date_col], errors='coerce')
            result_df = result_df.sort_values('sort_date').reset_index(drop=True)
            result_df = result_df.drop('sort_date', axis=1)
        else:
            # If no new records to add, just return existing data
            result_df = existing_df.copy()
    
    return result_df

async def cleanup_company_data(company: str):
    """Clean up processed data, cache, and models for a company"""
    try:
        api_logger.info(f"Cleaning up data for company: {company}")
        
        # 1. Delete processed data folder for this company
        if settings.USE_S3_STORAGE:
            # Delete from S3
            # List objects in processed folder for this company
            prefix = f"{settings.S3_PROCESSED_PREFIX}{company}/"
            
            api_logger.info(f"Listing objects in processed folder for this company ...")
            response = s3_service.s3_client.list_objects_v2(
                Bucket=s3_service.bucket_name,
                Prefix=prefix
            )
            
            #api_logger.info(f"Deleting objects in processed folder for this company ...")
            #if 'Contents' in response:
            #    for obj in response['Contents']:
            #    s3_service.s3_client.delete_object(
            #        Bucket=s3_service.bucket_name,
            #        Key=obj['Key']
            #    )
            #    api_logger.info(f"Deleted S3 object: {obj['Key']}")
                    
            # 2. Delete cache files for this company
            prefix = f"{settings.S3_CACHE_PREFIX}{company}/"
            
            api_logger.info(f"Listing objects in cache folder for this company ...")
            response = s3_service.s3_client.list_objects_v2(
                Bucket=s3_service.bucket_name,
                Prefix=prefix
            )
            
            api_logger.info(f"Deleting objects in cache folder for this company ...")
            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_service.s3_client.delete_object(
                        Bucket=s3_service.bucket_name,
                        Key=obj['Key']
                    )
                    api_logger.info(f"Deleted S3 cache object: {obj['Key']}")
            
            # 3. Delete model files for this company
            api_logger.info(f"Listing objects in model folder for this company ...")

            prefix = f"{settings.S3_MODEL_PREFIX}{company}"
            
            response = s3_service.s3_client.list_objects_v2(
                Bucket=s3_service.bucket_name,
                Prefix=prefix
            )
            
            api_logger.info(f"Deleting objects in model folder for this company ...")
            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_service.s3_client.delete_object(
                        Bucket=s3_service.bucket_name,
                        Key=obj['Key']
                    )
                    api_logger.info(f"Deleted S3 model object: {obj['Key']}")
            
            

            # 4. Delete events.json and weather.json from cache if they exist
            api_logger.info(f"Listing objects in cache folder for this company ...")
            for cache_file in ['events.json', 'weather_cache.json']:
                cache_key = f"{settings.S3_CACHE_PREFIX}{cache_file}"
                
                try:
                    s3_service.s3_client.head_object(
                        Bucket=s3_service.bucket_name,
                        Key=cache_key
                    )
                    
                    # If no exception, file exists - delete it
                    s3_service.s3_client.delete_object(
                        Bucket=s3_service.bucket_name,
                        Key=cache_key
                    )
                    api_logger.info(f"Deleted S3 cache file: {cache_key}")
                except:
                    # File doesn't exist - skip
                    pass
        else:
            # For local filesystem

            ## Remove the Directories for this company
            # 1. Delete processed data folder for this company
            processed_dir = os.path.join(settings.PROCESSED_DATA_DIR, company)
            if os.path.exists(processed_dir):
                shutil.rmtree(processed_dir)
                api_logger.info(f"Deleted local processed directory: {processed_dir}")
            
            model_dir = os.path.join(settings.MODELS_DIR, company)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
                api_logger.info(f"Deleted local model directory: {model_dir}")

            company_data_dir = os.path.join(settings.DATA_DIR, company)
            if os.path.exists(company_data_dir):
                shutil.rmtree(company_data_dir)
                api_logger.info(f"Deleted local data directory: {company_data_dir}")
            
            # Delete on base directory
            
            

            # 2. Delete cache files for this company
            cache_dir = os.path.join(settings.CACHE_DIR, company)
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                api_logger.info(f"Deleted local cache directory: {cache_dir}")
            
            # 3. Delete model files for this company
            for model_file in os.listdir(settings.MODELS_DIR):
                if company in model_file:
                    os.remove(os.path.join(settings.MODELS_DIR, model_file))
                    api_logger.info(f"Deleted local model file: {model_file}")
            
            # 4. Delete events.json and weather.json from cache if they exist
            for cache_file in ['events.json', 'weather_cache.json']:
                cache_path = os.path.join(settings.CACHE_DIR, cache_file)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    api_logger.info(f"Deleted local cache file: {cache_path}")

            # 5. Delete the models files for this company
            for model_file in os.listdir(settings.MODELS_DIR):
                if company in model_file:
                    os.remove(os.path.join(settings.MODELS_DIR, model_file))
                    api_logger.info(f"Deleted local model file: {model_file}")


        api_logger.info(f"Successfully cleaned up data for company: {company}")
        return True
    except Exception as e:
        api_logger.error(f"Error cleaning up data: {str(e)}")
        api_logger.error(traceback.format_exc())
        return False