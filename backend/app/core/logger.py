# app/core/logger.py
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path

# Create logs directory
logs_dir = Path(__file__).resolve().parent.parent.parent / "logs"
os.makedirs(logs_dir, exist_ok=True)

# Configure main logger
logger = logging.getLogger("sheffield_sales_forecast")
logger.setLevel(logging.INFO)

# Configure file handler for main logger
file_handler = RotatingFileHandler(
    logs_dir / "app.log",
    maxBytes=10485760,  # 10MB
    backupCount=5
)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)
logger.addHandler(file_handler)

# Configure console handler for main logger
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)
logger.addHandler(console_handler)

# Configure data processing logger
data_logger = logging.getLogger("sheffield_sales_forecast.data")
data_logger.setLevel(logging.INFO)

# Configure file handler for data logger
data_file_handler = RotatingFileHandler(
    logs_dir / "data.log",
    maxBytes=10485760,  # 10MB
    backupCount=5
)
data_file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)
data_logger.addHandler(data_file_handler)
data_logger.addHandler(console_handler)  # Use the same console handler

# Configure API logger
api_logger = logging.getLogger("sheffield_sales_forecast.api")
api_logger.setLevel(logging.INFO)

# Configure file handler for API logger
api_file_handler = RotatingFileHandler(
    logs_dir / "api.log",
    maxBytes=10485760,  # 10MB
    backupCount=5
)
api_file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)
api_logger.addHandler(api_file_handler)
api_logger.addHandler(console_handler) 