# app/services/s3_service.py
import os
import io
import boto3
import pandas as pd
from typing import Optional, Dict, List, Any, BinaryIO, Union
import traceback
from botocore.exceptions import ClientError

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from app.core.config import settings
from app.core.logger import data_logger

class S3Service:
    """Service for handling AWS S3 storage operations"""
    
    def __init__(self):
        """Initialize the S3 service with credentials from settings"""
        self.use_s3 = settings.USE_S3_STORAGE
        self.bucket_name = settings.AWS_S3_BUCKET_NAME
        
        if self.use_s3:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            data_logger.info(f"S3Service initialized for bucket: {self.bucket_name}")
        else:
            self.s3_client = None
            data_logger.info("S3Service initialized in local mode (not using S3)")
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3
        
        Args:
            s3_key: S3 key (path) of the file
            
        Returns:
            True if file exists, False otherwise
        """
        if not self.use_s3:
            return False
            
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                data_logger.error(f"Error checking if file exists in S3: {str(e)}")
                data_logger.error(traceback.format_exc())
                return False
    
    def list_objects(self, prefix: str) -> List[str]:
        """
        List objects in S3 with a specific prefix
        
        Args:
            prefix: S3 key prefix
            
        Returns:
            List of S3 keys (paths)
        """
        if not self.use_s3:
            return []
            
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [item['Key'] for item in response['Contents']]
            return []
            
        except Exception as e:
            data_logger.error(f"Error listing objects in S3: {str(e)}")
            data_logger.error(traceback.format_exc())
            return []
    
    def read_excel_file(self, s3_key: str) -> pd.DataFrame:
        """
        Read an Excel file from S3 into a pandas DataFrame
        
        Args:
            s3_key: S3 key (path) of the Excel file
            
        Returns:
            DataFrame with the Excel data
        """
        if not self.use_s3:
            return pd.DataFrame()
            
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            excel_data = response['Body'].read()
            
            return pd.read_excel(io.BytesIO(excel_data))
            
        except Exception as e:
            data_logger.error(f"Error reading Excel file from S3: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def read_csv_file(self, s3_key: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Read a CSV file from S3 into a pandas DataFrame
        
        Args:
            s3_key: S3 key (path) of the CSV file
            parse_dates: List of column names to parse as dates
            
        Returns:
            DataFrame with the CSV data
        """
        if not self.use_s3:
            return pd.DataFrame()
            
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            csv_data = response['Body'].read()
            
            # First read the CSV without date parsing to check columns
            df = pd.read_csv(io.BytesIO(csv_data))
            
            # Only parse dates if the specified columns exist
            if parse_dates:
                missing_date_cols = [col for col in parse_dates if col not in df.columns]
                if missing_date_cols:
                    data_logger.warning(f"Skipping date parsing for {s3_key} - missing columns: {missing_date_cols}")
                    return df
                else:
                    return pd.read_csv(io.BytesIO(csv_data), parse_dates=parse_dates)
            return df
            
        except Exception as e:
            data_logger.error(f"Error reading CSV file from S3: {str(e)}")
            data_logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def read_json_file(self, s3_key: str) -> Dict[str, Any]:
        """
        Read a JSON file from S3
        
        Args:
            s3_key: S3 key (path) of the JSON file
            
        Returns:
            Dictionary with the JSON data
        """
        if not self.use_s3:
            return {}
            
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            json_data = response['Body'].read()
            
            return pd.read_json(io.BytesIO(json_data), orient='records').to_dict('records')
            
        except Exception as e:
            data_logger.error(f"Error reading JSON file from S3: {str(e)}")
            data_logger.error(traceback.format_exc())
            return {}
    
    def write_csv_file(self, df: pd.DataFrame, s3_key: str) -> bool:
        """
        Write a pandas DataFrame to a CSV file in S3
        
        Args:
            df: DataFrame to write
            s3_key: S3 key (path) to write to
            
        Returns:
            Success flag
        """
        if not self.use_s3:
            return False
            
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            
            data_logger.info(f"Saved CSV file to S3: {s3_key}")
            return True
            
        except Exception as e:
            data_logger.error(f"Error writing CSV file to S3: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def write_json_file(self, data: Dict[str, Any], s3_key: str) -> bool:
        """
        Write a dictionary to a JSON file in S3
        
        Args:
            data: Dictionary to write
            s3_key: S3 key (path) to write to
            
        Returns:
            Success flag
        """
        if not self.use_s3:
            return False
            
        try:
            json_buffer = io.StringIO()
            pd.DataFrame(data).to_json(json_buffer, orient='records')
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_buffer.getvalue()
            )
            
            data_logger.info(f"Saved JSON file to S3: {s3_key}")
            return True
            
        except Exception as e:
            data_logger.error(f"Error writing JSON file to S3: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            s3_key: S3 key (path) of the file to delete
            
        Returns:
            Success flag
        """
        if not self.use_s3:
            return False
            
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            data_logger.info(f"Deleted file from S3: {s3_key}")
            return True
            
        except Exception as e:
            data_logger.error(f"Error deleting file from S3: {str(e)}")
            data_logger.error(traceback.format_exc())
            return False 
    
    def check_if_directory_exists(self, s3_key: str) -> bool:
        """
        Check if a directory exists in S3
        
        Args:
            s3_key: S3 key (path) of the directory
            
        Returns:
            True if directory exists, False otherwise
        """
        if not self.use_s3:
            return False
        
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                data_logger.error(f"Error checking if directory exists in S3: {str(e)}")
                data_logger.error(traceback.format_exc())
                return False    
        
