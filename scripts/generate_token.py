import json
import os
import webbrowser
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

CONFIG_PATH = "config/zerodha_token.json"
ENV_PATH = ".env"

# Load existing .env variables
load_dotenv(ENV_PATH)

# Load Zerodha config
with open(CONFIG_PATH) as f:
    config = json.load(f)

API_KEY = config["api_key"]
API_SECRET = config["api_secret"]

# Initialize Kite and open login URL
kite = KiteConnect(api_key=API_KEY)
print("➡️ Login URL:", kite.login_url())
webbrowser.open(kite.login_url())

# Input and generate access token
request_token = input("🔐 Paste request_token here: ")
data = kite.generate_session(request_token, api_secret=API_SECRET)

# Save to zerodha_token.json
config["access_token"] = data["access_token"]
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)
print("✅ Access token updated in zerodha_token.json")

# Save to .env
set_key(ENV_PATH, "ZERODHA_ACCESS_TOKEN", data["access_token"])
print("✅ Access token updated in .env as ZERODHA_ACCESS_TOKEN")
