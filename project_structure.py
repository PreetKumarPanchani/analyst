import os

def create_structure(base_path, structure):
    for key, value in structure.items():
        path = os.path.join(base_path, key)
        if isinstance(value, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, value)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("")

# Define the project structure
structure = {
    "sheffield_sales_forecast": {
        "backend": {
            "app": {
                "api": {
                    "__init__.py": "",
                    "routes": {
                        "__init__.py": "",
                        "sales.py": "",
                        "forecasts.py": "",
                        "events.py": "",
                    },
                },
                "core": {
                    "__init__.py": "",
                    "config.py": "",
                    "logger.py": "",
                },
                "data": {
                    "__init__.py": "",
                    "loader.py": "",
                    "processor.py": "",
                },
                "models": {
                    "__init__.py": "",
                    "prophet_model.py": "",
                },
                "services": {
                    "__init__.py": "",
                    "events_service.py": "",
                    "weather_service.py": "",
                    "forecast_service.py": "",
                },
                "main.py": "",
            },
            "data": {
                "raw": {},
                "processed": {},
                "cache": {},
                "models": {},
            },
            "requirements.txt": "",
        },
        "frontend": {
            "public": {},
            "src": {
                "components": {
                    "CompanySelector.jsx": "",
                    "ForecastChart.jsx": "",
                    "ForecastControls.jsx": "",
                    "ForecastTypeSelector.jsx": "",
                    "InsightsPanel.jsx": "",
                },
                "pages": {
                    "_app.js": "",
                    "index.js": "",
                    "api": {},
                    "forecast": {
                        "[company]": {
                            "[type]": {
                                "[target].js": "",
                            },
                        },
                    },
                },
                "styles": {},
            },
            "package.json": "",
            "next.config.js": "",
        },
        "README.md": "",
    }
}

base_directory = os.getcwd()
create_structure(base_directory, structure)

print("Project structure created successfully!")
