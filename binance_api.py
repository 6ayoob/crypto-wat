import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from config import BINANCE_API_KEY, BINANCE_API_SECRET

BASE_URL = "https://api.binance.com"

def send_signed_request(http_method, url_path, payload={}):
    timestamp = int(time.time() * 1000)
    payload.update({"timestamp": timestamp})
    query_string = urlencode(payload)
    signature = hmac.new(BINANCE_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE_URL}{url_path}?{query_string}&signature={signature}"
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    if http_method == "GET":
        response = requests.get(url, headers=headers)
    elif http_method == "POST":
        response = requests.post(url, headers=headers)
    else:
        raise Exception("Unsupported HTTP method")
    return response.json()

def get_price(symbol):
    url = f"{BASE_URL}/api/v3/ticker/price"
    response = requests.get(url, params={"symbol": symbol})
    data = response.json()
    return float(data.get("price", 0))

def get_klines(symbol, interval="1h", limit=200):
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    return response.json()

def get_balance(asset="USDT"):
    data = send_signed_request("GET", "/api/v3/account")
    for b in data.get("balances", []):
        if b["asset"] == asset:
            return float(b["free"])
    return 0.0

def place_order(symbol, side, quantity):
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
    }
    return send_signed_request("POST", "/api/v3/order", params)
