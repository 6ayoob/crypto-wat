import requests
import time
import hmac
import hashlib
import base64
import json

# بيانات API الخاصة بك (لا تشاركها مع أحد)
API_KEY = "6e2d2b3f-636a-424a-a97e-5154e39e525a"
SECRET_KEY = "D4B9966385BEE5A7B7D8791BA5C0539F"
PASSPHRASE = "Ta123456&"

BASE_URL = "https://www.okx.com"

def _get_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

def _sign(message: str, secret_key: str):
    mac = hmac.new(secret_key.encode('utf-8'), message.encode('utf-8'), hashlib.sha256)
    d = mac.digest()
    return base64.b64encode(d).decode()

def _get_headers(method, request_path, body=""):
    timestamp = _get_timestamp()
    if body:
        body_str = json.dumps(body)
    else:
        body_str = ""
    message = timestamp + method.upper() + request_path + body_str
    sign = _sign(message, SECRET_KEY)
    return {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }

def get_last_price(symbol):
    """جلب السعر الأخير لزوج التداول"""
    path = f"/api/v5/market/ticker?instId={symbol.replace('/', '-')}"
    url = BASE_URL + path
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            return float(data["data"][0]["last"])
    return None

def place_limit_order(symbol, side, price, size):
    """تنفيذ أمر محدود Limit Order"""
    path = "/api/v5/trade/order"
    url = BASE_URL + path
    body = {
        "instId": symbol.replace("/", "-"),
        "tdMode": "cash",   # أو "cross" أو "isolated" حسب حسابك
        "side": side,       # "buy" أو "sell"
        "ordType": "limit",
        "px": str(price),
        "sz": str(size)
    }
    headers = _get_headers("POST", path, body)
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return response.json()
    return None

def place_market_order(symbol, side, size):
    """تنفيذ أمر سوق Market Order"""
    path = "/api/v5/trade/order"
    url = BASE_URL + path
    body = {
        "instId": symbol.replace("/", "-"),
        "tdMode": "cash",
        "side": side,
        "ordType": "market",
        "sz": str(size)
    }
    headers = _get_headers("POST", path, body)
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return response.json()
    return None

def get_historical_candles(symbol, bar="1H", limit=100):
    """جلب بيانات الشموع (الكندلز) التاريخية"""
    path = f"/api/v5/market/candles?instId={symbol.replace('/', '-')}&bar={bar}&limit={limit}"
    url = BASE_URL + path
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "data" in data:
            # البيانات مرتبة من الأحدث للأقدم في OKX API
            return data["data"][::-1]  # نعكسها لتكون من الأقدم للأحدث
    return None
