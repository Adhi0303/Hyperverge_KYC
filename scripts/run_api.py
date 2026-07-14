"""
scripts/run_api.py
==================
Convenience launcher for the FastAPI backend.

Usage:
    .venv\\Scripts\\python.exe scripts/run_api.py
"""

import sys, os, subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

subprocess.run([
    sys.executable, "-m", "uvicorn",
    "backend.main:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--log-level", "info",
    # NOTE: No --reload on Windows — it breaks GPU model loading
], check=True)
