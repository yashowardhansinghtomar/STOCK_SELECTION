import json
import os

with open("config/zerodha_token.json") as f:
    config = json.load(f)

_API_KEY = config["api_key"]
_ACCESS_TOKEN = config["access_token"]

from kiteconnect import KiteConnect, KiteTicker

_kite = None
_ticker = None

def get_kite():
    global _kite
    if _kite is None:
        _kite = KiteConnect(api_key=_API_KEY)
        _kite.set_access_token(_ACCESS_TOKEN)
    return _kite

def get_ticker():
    global _ticker
    if _ticker is None:
        _ticker = KiteTicker(api_key=_API_KEY, access_token=_ACCESS_TOKEN)
    return _ticker
