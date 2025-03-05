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

structure = {
    "sales_analytics": {
        "backend": {
            "app": {
                "__init__.py": "",
                "main.py": "",
                "models": {
                    "__init__.py": "",
                    "schemas.py": "",
                },
                "routers": {
                    "__init__.py": "",
                    "sales.py": "",
                    "forecast.py": "",
                },
                "services": {
                    "__init__.py": "",
                    "data_service.py": "",
                    "forecast.py": "",
                },
                "utils": {
                    "__init__.py": "",
                    "excel_utils.py": "",
                },
            },
            "tests": {
                "__init__.py": "",
                "test_data_service.py": "",
                "test_forecast.py": "",
            },
            "requirements.txt": "",
        },
        "frontend": {
            "app": {
                "page.tsx": "",
                "layout.tsx": "",
                "globals.css": "",
                "dashboard": {
                    "page.tsx": "",
                },
                "api": {},
            },
            "components": {
                "ui": {},
                "charts": {},
                "dashboard": {},
            },
            "lib": {
                "api.ts": "",
                "utils.ts": "",
            },
            "public": {},
            ".env.local": "",
            "package.json": "",
            "next.config.js": "",
            "tsconfig.json": "",
        },
    }
}

base_directory = os.getcwd()
create_structure(base_directory, structure)

print("Project structure created successfully!")
