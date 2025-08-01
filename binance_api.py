import requests
import time
import hmac
import hashlib

API_KEY = "kLUkQ10bjOQkAD9I2xNIUOxLRiBWFRDmw2RJXHMb6sodChOTEfCmaBg77jpBUCG4"
API_SECRET = "TuVFC67mDsRoVRmVE9rPb0qfCEdMinnKjRrYZO3pkXVk7m12ZjDvNvXKYKgQgPVo"

BASE_URL = "https://api.binance.com"

def get_server_time():
    try:
        response = requests.get(BASE_URL + "/api/v3/time")
        response.raise_for_status()
        return response.json()['serverTime']
    except Exception as e:
        print(f"❌ خطأ في جلب توقيت السيرفر: {e}")
        return int(time.time() * 1000)  # fallback للتوقيت المحلي

def get_timestamp():
    return get_server_time()

def get_balance(symbol="USDT"):
    endpoint = "/api/v3/account"
    url = BASE_URL + endpoint

    timestamp = get_timestamp()
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(
        API_SECRET.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-MBX-APIKEY": API_KEY
    }

    try:
        response = requests.get(f"{url}?{query_string}&signature={signature}", headers=headers)
        if response.status_code != 200:
            print(f"❌ فشل في الاتصال بـ Binance: {response.status_code}")
            print("الرد:", response.text)
            return 0.0

        data = response.json()
        for asset in data["balances"]:
            if asset["asset"] == symbol:
                return float(asset["free"])
        print(f"⚠️ لم يتم العثور على رصيد لـ {symbol}")
        return 0.0

    except Exception as e:
        print("❌ خطأ في get_balance:", e)
        return 0.0
