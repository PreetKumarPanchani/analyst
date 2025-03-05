# Sales Analytics

A full-stack application for analyzing sales data and forecasting future sales using AI. The application consists of a FastAPI backend for data processing and a Next.js frontend for visualization.

## Features

- **Sales Analysis**: Analyze sales data by time period, product, category, and register
- **Sales Forecasting**: Predict future sales using Facebook Prophet time series forecasting
- **Interactive Visualizations**: Dynamic charts and graphs to explore sales data
- **Product Analysis**: Identify top-selling products and categories
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

The application is built with a modern tech stack:

- **Backend**: 
  - FastAPI for RESTful API
  - Pandas for data processing
  - Prophet for time series forecasting
  - Python 3.10+

- **Frontend**:
  - Next.js 14+ for React framework
  - Recharts for interactive charts
  - Tailwind CSS for styling
  - TypeScript for type safety

## Project Structure

```
sales-analytics/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application
│   │   ├── models/             # Pydantic data models
│   │   │   ├── __init__.py
│   │   │   └── schemas.py
│   │   ├── routers/            # API routes
│   │   │   ├── __init__.py
│   │   │   ├── sales.py
│   │   │   └── forecast.py
│   │   ├── services/           # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── data_service.py # Data loading and processing
│   │   │   └── forecast.py     # Prophet forecasting
│   │   └── utils/              # Utility functions
│   │       ├── __init__.py
│   │       └── excel_utils.py
│   ├── tests/                  # Test files
│   │   ├── __init__.py
│   │   ├── test_data_service.py
│   │   └── test_forecast.py
│   └── requirements.txt        # Python dependencies
└── frontend/
    ├── app/                    # Next.js app directory
    │   ├── page.tsx            # Home page
    │   ├── layout.tsx          # Root layout
    │   ├── globals.css         # Global styles
    │   ├── dashboard/          # Dashboard page
    │   │   └── page.tsx
    │   └── api/                # API routes (if needed)
    ├── components/             # React components
    │   ├── ui/                 # UI components
    │   ├── charts/             # Chart components
    │   └── dashboard/          # Dashboard components
    ├── lib/                    # Utility functions
    │   ├── api.ts              # API client
    │   └── utils.ts            # Utility functions
    ├── public/                 # Static assets
    ├── .env.local              # Environment variables
    ├── package.json            # JS dependencies
    ├── next.config.js          # Next.js configuration
    └── tsconfig.json           # TypeScript configuration

```

## Getting Started

### Prerequisites

- Docker and Docker Compose (for containerized setup)
- Python 3.10+ (for local backend development)

### Running with Docker

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sales-analytics.git
   cd sales-analytics
   ```

2. Place your Excel data files in the `data/` directory:
   ```bash
   mkdir -p data
   # Copy your Excel files to the data directory
   ```

3. Start the application with Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Local Development

#### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   pip install uv
   uv venv --python 3.10
   source .venv/Scripts/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI application:
   ```bash
   uvicorn app.main:app --reload
   ```

#### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set environment variables:
   ```bash
   echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api" > .env.local
   ```

4. Run the Next.js development server:
   ```bash
   npm run dev
   ```

## Data Format

The application expects Excel files with the following structure:

1. File naming convention:
   - `forge_XX_export_transactions_report_*.xlsx` for Forge Bakehouse data
   - `cpl_XX_export_transactions_report_*.xlsx` for CPLGPOPUP Ltd data
   - Where `XX` represents the month range (e.g., `12` for January-February)

2. Each Excel file should contain at least these sheets:
   - `Sales`: Contains transaction data
   - `Sales Items`: Contains item-level data for each transaction

## Adding Your Own Data

1. Ensure your Excel files follow the expected format (see above)
2. Place the files in the `data/` directory
3. Restart the application if running in Docker, or reload the page if running locally

## License

[MIT License](LICENSE)

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) - Fast API framework
- [Prophet](https://facebook.github.io/prophet/) - Time series forecasting
- [Next.js](https://nextjs.org/) - React framework
- [Recharts](https://recharts.org/) - Charting library
- [Tailwind CSS](https://tailwindcss.com/) - CSS framework