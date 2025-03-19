# Use Python 3.9 slim image for a smaller container size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
# This will respect .dockerignore (which we'll create next)
COPY . .

# Create directories that are referenced in the code even though they are in .dockerignore
RUN mkdir -p data/cache data/models data/forge data/cpl debug 

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]