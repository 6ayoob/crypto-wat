import time
import hmac
import hashlib
import base64
import json
import requests
from urllib.parse import urlencode

# بيانات OKX الحقيقية
OKX_API_KEY = "6e2d2b3f-636a-424a-a97e-5154e39e525a"
OKX_SECRET_KEY = "D4B9966385BEE5A7B7D8791BA5C0539F"
OKX_PASSPHRASE = "Ta123456&"

BASE_URL = "https://www.okx.com"

def okx_request(method, endpoint, params=None, data=None):
from datetime import datetime

timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
    request_path = endpoint

    if params:
        query_string = urlencode(params)
        request_path += '?' + query_string
    body = json.dumps(data) if data else ""

    prehash = f"{timestamp}{method}{request_path}{body}"
    mac = hmac.new(OKX_SECRET_KEY.encode(), prehash.encode(), hashlib.sha256)
    sign = base64.b64encode(mac.digest()).decode()

    headers = {
        "Content-Type": "application/json",
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
        "x-simulated-trading": "0"
    }

    try:
        response = requests.request(method, BASE_URL + endpoint, headers=headers, params=params, data=body)
        result = response.json()
        if result.get("code") != "0":
            print(f"❌ OKX Error: {result}")
        return result
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

# مثال: جلب رصيد USDT
def get_usdt_balance():
    result = okx_request("GET", "/api/v5/account/balance", params={"ccy": "USDT"})
    try:
        if result and result.get("code") == "0":
            balance = result["data"][0]["details"][0]["cashBal"]
            print(f"✅ USDT Balance: {balance}")
            return float(balance)
    except Exception as e:
        print(f"❌ Balance error: {e}")
    return 0.0

# استراتيجية بسيطة للتشغيل
def run_strategy():
    print("🚀 Running strategy...")
    balance = get_usdt_balance()
    if balance > 0:
        print(f"🎯 لديك {balance} USDT متاحة.")
    else:
        print("⚠️ لا يوجد رصيد أو يوجد خطأ.")

# بدء التنفيذ
if __name__ == "__main__":
    run_strategy()
