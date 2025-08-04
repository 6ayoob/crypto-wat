import time
import hmac
import hashlib
import base64
import requests
import json
from datetime import datetime

# مفاتيح حساب OKX (حقيقية حسب ما زودتني)
OKX_API_KEY = "f267dba2-ece2-4a9a-aa11-f7a10579be44"
OKX_SECRET_KEY = "022CBE3D1590783852B64002F97FD9BD"
OKX_PASSPHRASE = "Ta123456&"

BASE_URL = "https://www.okx.com"

# إرسال طلب موقع لـ OKX
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

    body = json.dumps(data) if data and method != "GET" else ""
    message = f"{timestamp}{method}{request_path}{body}"
    
    sign = base64.b64encode(
        hmac.new(
            OKX_SECRET_KEY.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
    ).decode()

    headers = {
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
        "Content-Type": "application/json"
    }

    url = BASE_URL + endpoint
    try:
        response = requests.request(method, url, headers=headers, params=params if method=="GET" else None, data=body if method!="GET" else None)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None

# استراتيجية بسيطة لعرض الرصيد
def run_strategy():
    print("🚀 Running strategy...")
    balance_data = okx_request("GET", "/api/v5/account/balance")
    if balance_data and "data" in balance_data:
        currencies = balance_data["data"][0].get("details", [])
        if currencies:
            for asset in currencies:
                ccy = asset.get("ccy")
                avail = asset.get("availBal")
                if float(avail) > 0:
                    print(f"💰 {ccy}: {avail}")
        else:
            print("⚠️ لا يوجد رصيد.")
    else:
        print(f"❌ OKX Error: {balance_data if balance_data else 'No response'}")

# تنفيذ الاستراتيجية
if __name__ == "__main__":
    run_strategy()
