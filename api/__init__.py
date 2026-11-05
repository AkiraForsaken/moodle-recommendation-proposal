"""FastAPI bridge between the Moodle plugins and the Python analytics backend.

Run (from the project root, where model/, recommender/ and data/ live):

    API_TOKEN=<secret> uvicorn api:create_app --factory --host 0.0.0.0 --port 8010
"""

from .main import create_app

__all__ = ["create_app"]
