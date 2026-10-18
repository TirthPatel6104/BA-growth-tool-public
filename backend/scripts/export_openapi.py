"""
Export OpenAPI JSON Specification
Generates documentation/openapi.json for API documentation and Postman import.
"""

import os
import json
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from main import app

def export_openapi_spec():
    doc_dir = os.path.join(os.path.dirname(_BACKEND_DIR), "documentation")
    os.makedirs(doc_dir, exist_ok=True)
    
    openapi_path = os.path.join(doc_dir, "openapi.json")
    spec = app.openapi()
    
    with open(openapi_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
        
    print(f"Exported OpenAPI specification to: {openapi_path}")

if __name__ == "__main__":
    export_openapi_spec()
