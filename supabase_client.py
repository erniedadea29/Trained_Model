"""Shared Supabase connection, configured from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv(Path(__file__).parent / ".env")

BUCKET = "models"
MODEL_NAME = "budget_estimator"


def get_client() -> Client:
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key or "YOUR-PROJECT-REF" in url:
        raise SystemExit("Missing Supabase credentials: copy .env.example to .env and fill it in.")
    return create_client(url, key)
