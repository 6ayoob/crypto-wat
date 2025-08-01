import requests
import time
import hmac
import hashlib

# مفاتيح API الخاصة بك
API_KEY = "kLUkQ10bjOQkAD9I2xNIUOxLRiBWFRDmw2RJXHMb6sodChOTEfCmaBg77jpBUCG4"
API_SECRET = "TuVFC67mDsRoVRmVE9rPb0qfCEdMinnKjRrYZO3pkXVk7m12ZjDvNvXKYKgQgPVo"

BASE_URL = "https://api.binance.com"

def get_timestamp():
    return int(time.time() * 1000)

def get_balance(symbol="USDT"):
    """جلب رصيد العملة المطلوبة"""
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
