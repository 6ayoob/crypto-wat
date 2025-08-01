import requests
import time
import hmac
import hashlib

API_KEY = "kLUkQ10bjOQkAD9I2xNIUOxLRiBWFRDmw2RJXHMb6sodChOTEfCmaBg77jpBUCG4"
API_SECRET = "TuVFC67mDsRoVRmVE9rPb0qfCEdMinnKjRrYZO3pkXVk7m12ZjDvNvXKYKgQgPVo"

BASE_URL = "https://api.binance.com"

def get_server_time():
    try:
        response = requests.get(f"{BASE_URL}/api/v3/time")
        response.raise_for_status()
        return response.json()['serverTime']
    except Exception as e:
        print(f"❌ خطأ في جلب توقيت السيرفر: {e}")
        return int(time.time() * 1000)  # fallback للتوقيت المحلي

def sign_request(params):
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def get_balance(symbol="USDT"):
    endpoint = "/api/v3/account"
    url = BASE_URL + endpoint
    timestamp = get_server_time()

    params = {
        "timestamp": timestamp
    }
    signature = sign_request(params)

    headers = {
        "X-MBX-APIKEY": API_KEY
    }

    try:
        response = requests.get(f"{url}?timestamp={timestamp}&signature={signature}", headers=headers)
        response.raise_for_status()
        data = response.json()
        for asset in data.get("balances", []):
            if asset["asset"] == symbol:
                return float(asset["free"])
        print(f"⚠️ لم يتم العثور على رصيد لـ {symbol}")
        return 0.0
    except Exception as e:
        print(f"❌ خطأ في get_balance: {e}")
        return 0.0

def place_order(symbol, side, quantity):
    endpoint = "/api/v3/order"
    url = BASE_URL + endpoint
    timestamp = get_server_time()

    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": timestamp
    }
    signature = sign_request(params)

    headers = {
        "X-MBX-APIKEY": API_KEY
    }

    try:
        response = requests.post(f"{url}?symbol={symbol}&side={side}&type=MARKET&quantity={quantity}&timestamp={timestamp}&signature={signature}", headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ خطأ في تنفيذ أمر {side} لـ {symbol}: {e}")
        return None

def get_klines(symbol, interval="15m", limit=50):
    url = f"{BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات {symbol}: {e}")
        return []
