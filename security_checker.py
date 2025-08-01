import time
import hmac
import hashlib
import requests

# 🔹 ضع مفاتيح Binance الخاصة بك هنا
API_KEY = "TuVFC67mDsRoVRmVE9rPb0qfCEdMinnKjRrYZO3pkXVk7m12ZjDvNvXKYKgQgPVo"
API_SECRET = "TuVFC67mDsRoVRmVE9rPb0qfCEdMinnKjRrYZO3pkXVk7m12ZjDvNvXKYKgQgPVo"

BASE_URL = "https://api.binance.com"

def sign(params, secret):
    query = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    return query + "&signature=" + signature

def check_api_security():
    endpoint = "/api/v3/account"
    timestamp = int(time.time() * 1000)
    params = {"timestamp": timestamp, "recvWindow": 10000}
    
    query = sign(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}

    print("🔍 فحص إعدادات Binance API ...")
    
    try:
        response = requests.get(BASE_URL + endpoint + "?" + query, headers=headers)
        
        if response.status_code == 200:
            print("✅ الاتصال ناجح! المفتاح يعمل بشكل سليم.")
            print("ℹ️ إذا لم تكن قد قيدت الـ IP، ننصحك بإضافته في Binance API Management.")
            return response.json()

        elif response.status_code == 451:
            print("❌ خطأ 451: Binance رفض الطلب لأسباب أمنية.")
            print("💡 الحل: فعّل IP Whitelist للمفتاح من إعدادات Binance.")
        
        elif response.status_code == 401:
            print("❌ خطأ 401: التوقيع أو المفاتيح غير صحيحة.")
            print("💡 تحقق من API_KEY و API_SECRET.")
        
        else:
            print(f"❌ خطأ غير متوقع: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"⚠️ خطأ في الاتصال: {e}")

# ✅ تشغيل الفحص
if __name__ == "__main__":
    check_api_security()
