import time
import hmac
import hashlib
import base64
import requests
import json
from datetime import datetime

OKX_API_KEY = "6e2d2b3f-636a-424a-a97e-5154e39e525a"
OKX_SECRET_KEY = "D4B9966385BEE5A7B7D8791BA5C0539F"
OKX_PASSPHRASE = "Ta123456&"
BASE_URL = "https://www.okx.com"

def generate_signature(timestamp, method, request_path, body=""):
    message = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(OKX_SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def okx_request(method, endpoint, params=None, data=None):
    timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
    if params is None:
        params = {}
    if data is None:
        data = {}

    request_path = endpoint
    if method == "GET" and params:
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        request_path += '?' + query_string

    body_str = ""
    if method != "GET" and data:
        body_str = json.dumps(data)
    sign = generate_signature(timestamp, method, request_path, body_str)

    headers = {
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
        "Content-Type": "application/json"
    }

    url = BASE_URL + endpoint
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, params=params)
        else:
            resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ OKX API error: {e}")
        return None

def get_balance(ccy="USDT"):
    data = okx_request("GET", "/api/v5/account/balance", params={"ccy": ccy})
    try:
        if data and "data" in data:
            for item in data["data"]:
                if item["ccy"] == ccy:
                    return float(item["details"][0]["availBal"])
        return 0.0
    except Exception:
        return 0.0

def get_last_price(instId):
    data = okx_request("GET", "/api/v5/market/ticker", params={"instId": instId})
    try:
        if data and "data" in data:
            return float(data["data"][0]["last"])
        return None
    except Exception:
        return None

def get_historical_candles(instId, bar="1D", limit=30):
    data = okx_request("GET", "/api/v5/market/candles", params={"instId": instId, "bar": bar, "limit": limit})
    if data and "data" in data:
        return data["data"]
    return None

def place_limit_order(instId, side, price, size):
    body = {
        "instId": instId,
        "tdMode": "cash",
        "side": side,
        "ordType": "limit",
        "px": str(price),
        "sz": str(size)
    }
    return okx_request("POST", "/api/v5/trade/order", data=body)

def place_market_order(instId, side, size):
    body = {
        "instId": instId,
        "tdMode": "cash",
        "side": side,
        "ordType": "market",
        "sz": str(size)
    }
    return okx_request("POST", "/api/v5/trade/order", data=body)
