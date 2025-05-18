import json
import os
import webbrowser
from kiteconnect import KiteConnect

CONFIG_PATH = "config/zerodha_token.json"

with open(CONFIG_PATH) as f:
    config = json.load(f)

API_KEY = config["api_key"]
API_SECRET = config["api_secret"]

kite = KiteConnect(api_key=API_KEY)
print("➡️ Login URL:", kite.login_url())
webbrowser.open(kite.login_url())

request_token = input("🔐 Paste request_token here: ")
data = kite.generate_session(request_token, api_secret=API_SECRET)

# Save access_token back
config["access_token"] = data["access_token"]
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print("✅ Access token updated in zerodha_token.json")
