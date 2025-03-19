# Sales Forecast Application

A sales forecasting application for analyzing and predicting sales data for Sheffield companies using historical sales data, weather and events data. The system provides forecasting capabilities for revenue, products, and categories with interactive visualizations.

## Features

- Revenue forecasting with time series analysis 
- Product-level sales forecasting
- Category-based forecasting
- Interactive dashboards with charts and metrics
- Weather and events impact

## Tech Stack

- **Backend**: FastAPI, Pandas, Facebook Prophet Model 
- **Frontend**: Next.js, React, Recharts, TailwindCSS
- **Data Processing**: Pandas, NumPy, 

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
   ```bash
   pip install uv 
   uv venv --python 3.10
   source .venv/Scripts/activate
   # or on Windows
   .venv\Scripts\activate
   ```

2. Navigate to the backend directory:
   ```bash
   cd backend
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Run the FastAPI application:
   ```bash
   uvicorn app.main:app --reload --port 8001
   ```

5. Run a python file (alternative to running the FastAPI application):
    ```bash
    python app/main.py
    ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

## Usage

Once both servers are running:

1. Access the frontend at: http://localhost:3000
2. Access the API documentation at: http://localhost:8001/api/v1/docs

## Data Structure

The application uses data files from two companies (Forge and CPL):
- Sales data
- Sales Items data
- Sales Payments data
- Deleted Sales Items data



