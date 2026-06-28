"""Shared test configuration for CausalForge backend tests."""
import os

# Set env vars before any test imports (main.py checks these at module level)
os.environ.setdefault("NIM_API_KEY", "test-key-for-testing")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB", "causalforge_test")
