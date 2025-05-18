# core/token_manager.py

import os
import json
from pathlib import Path

TOKEN_PATH = Path("project_data/secrets/zerodha_token.json")
TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

def save_access_token(data):
    with open(TOKEN_PATH, "w") as f:
        json.dump(data, f)

def load_access_token():
    if not TOKEN_PATH.exists():
        raise FileNotFoundError("Zerodha access token not found. Please generate it first.")
    with open(TOKEN_PATH, "r") as f:
        return json.load(f)

def get_saved_access_token():
    return load_access_token()["access_token"]
