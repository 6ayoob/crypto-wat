import time
import requests
import hmac
import hashlib
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API, BINANCE_SECRET

open_trades = []
last_report_time = 0

# ✅ قائمة Endpoints احتياطية
ENDPOINTS = [
    "https://api.binance.com",
    "https://api-gateway.binance.com",
    "https://binance-proxy.cloudflarest.workers.dev"  # يمكنك إضافة Proxy خاص بك
]

current_base = ENDPOINTS[0]

# ✅ إرسال Telegram
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except:
        print("⚠️ فشل إرسال التنبيه إلى Telegram")

# ✅ تبديل الـ Endpoint عند الفشل
def switch_endpoint():
    global current_base
    idx = ENDPOINTS.index(current_base)
    new_idx = (idx + 1) % len(ENDPOINTS)
    current_base = ENDPOINTS[new_idx]
    send_telegram(f"🔄 تبديل Endpoint إلى: {current_base}")

# ✅ طلب API مع إعادة محاولة + تبديل Endpoint عند الفشل
def api_request(method, path, headers=None, params=None):
    global current_base
    for attempt in range(3):
        try:
            url = f"{current_base}{path}"
            if method == "GET":
                r = requests.get(url, headers=headers, params=params, timeout=5)
            else:
                r = requests.post(url, headers=headers, params=params, timeout=5)

            if r.status_code == 200:
                return r
            elif r.status_code == 451:
                send_telegram("❌ خطأ 451: IP غير مسموح - يجب تفعيل IP Whitelist.")
                return r
            else:
                send_telegram(f"⚠️ خطأ {r.status_code}: {r.text}")
        except Exception as e:
            send_telegram(f"⚠️ فشل الاتصال بمحاولة {attempt+1}: {e}")

        # إذا فشل → تبديل Endpoint
        switch_endpoint()
        time.sleep(2)
    return None

# ✅ جلب IP الخارجي
def get_external_ip():
    try:
        return requests.get("https://api.ipify.org?format=json", timeout=5).json().get("ip", "غير معروف")
    except:
        return "غير معروف"

# ✅ جلب توقيت Binance مع محاولات
def get_binance_server_time():
    r = api_request("GET", "/api/v3/time")
    if r and r.status_code == 200:
        return r.json()['serverTime']
    raise SystemExit("⛔ فشل جلب توقيت Binance من جميع الخوادم.")

# ✅ توقيع HMAC
def sign_params(params):
    query = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query + "&signature=" + signature

# ✅ فحص حالة API
def check_api_status():
    ts = get_binance_server_time()
    params = {"timestamp": ts, "recvWindow": 10000}
    headers = {"X-MBX-APIKEY": BINANCE_API}
    query = sign_params(params)
    r = api_request("GET", f"/api/v3/account?{query}", headers=headers)
    if not r: return False
    if r.status_code == 200:
        return True
    return False

# ✅ جلب الرصيد
def get_balance(asset="USDT"):
    ts = get_binance_server_time()
    params = {"timestamp": ts, "recvWindow": 10000}
    headers = {"X-MBX-APIKEY": BINANCE_API}
    query = sign_params(params)
    r = api_request("GET", f"/api/v3/account?{query}", headers=headers)
    if not r or r.status_code != 200: return 0.0
    balances = r.json().get("balances", [])
    return float(next((b["free"] for b in balances if b["asset"] == asset), 0.0))

# ✅ جلب بيانات الأسعار
def get_klines(symbol):
    r = api_request("GET", "/api/v3/klines", params={"symbol": symbol, "interval": "15m", "limit": 50})
    return r.json() if r and r.status_code == 200 else []

# ✅ حساب EMA
def calculate_ema(prices, length):
    if len(prices) < length: return None
    k, ema = 2/(length+1), prices[0]
    for p in prices[1:]: ema = p*k + ema*(1-k)
    return ema

# ✅ تنفيذ أوامر شراء/بيع
def place_order(symbol, side, qty):
    ts = get_binance_server_time()
    params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty, "timestamp": ts, "recvWindow": 10000}
    headers = {"X-MBX-APIKEY": BINANCE_API}
    query = sign_params(params)
    r = api_request("POST", f"/api/v3/order?{query}", headers=headers)
    return r.json() if r and r.status_code == 200 else None

# ✅ منطق التداول
def trade_logic():
    global open_trades
    try:
        with open("coins.txt") as f:
            coins = [x.strip().upper() for x in f if x.strip()]
    except:
        send_telegram("❌ ملف coins.txt غير موجود")
        return

    usdt = get_balance()
    if usdt < 30:
        send_telegram(f"⚠️ رصيد غير كافي: {usdt:.2f} USDT")
        return

    send_telegram(f"📊 فحص السوق لـ {len(coins)} عملة - رصيد {usdt:.2f} USDT")

    for c in coins:
        pair = c + "USDT"
        data = get_klines(pair)
        if len(data) < 20: continue
        closes = [float(k[4]) for k in data]
        ema9, ema21 = calculate_ema(closes[-20:], 9), calculate_ema(closes[-20:], 21)
        if ema9 and ema21 and ema9 > ema21 and not any(t["symbol_pair"] == pair for t in open_trades):
            qty = round(30/closes[-1], 5)
            order = place_order(pair, "BUY", qty)
            if order:
                send_telegram(f"✅ شراء {c} بسعر {closes[-1]:.4f}, كمية {qty}")
                open_trades.append({"symbol_pair": pair, "symbol": c, "qty": qty, "entry_price": closes[-1],
                                    "target1": closes[-1]*1.05, "target2": closes[-1]*1.10, "stop_loss": closes[-1]*0.98,
                                    "sold_target1": False})
        time.sleep(1)

# ✅ متابعة الصفقات
def follow_trades():
    global open_trades
    still = []
    for t in open_trades:
        data = get_klines(t["symbol_pair"])
        if not data: continue
        price = float(data[-1][4])
        if not t["sold_target1"] and price >= t["target1"]:
            half = round(t["qty"]/2,5)
            place_order(t["symbol_pair"], "SELL", half)
            send_telegram(f"🎯 بيع نصف {t['symbol']} عند +5% بسعر {price:.4f}")
            t["sold_target1"] = True; t["qty"] -= half
        elif price >= t["target2"]:
            place_order(t["symbol_pair"], "SELL", t["qty"])
            send_telegram(f"🏁 بيع كامل {t['symbol']} عند +10% بسعر {price:.4f}")
        elif price <= t["stop_loss"]:
            place_order(t["symbol_pair"], "SELL", t["qty"])
            send_telegram(f"🚨 وقف خسارة {t['symbol']} عند {price:.4f}")
        else:
            still.append(t)
    open_trades = still

# ✅ تقرير دوري
def send_status():
    global last_report_time
    if time.time() - last_report_time > 3600:
        send_telegram(f"📡 تقرير: IP {get_external_ip()} | صفقات {len(open_trades)} | Endpoint {current_base}")
        last_report_time = time.time()

# ✅ التشغيل
if __name__ == "__main__":
    send_telegram(f"🤖 تشغيل البوت - IP {get_external_ip()} - Endpoint {current_base}")
    if not check_api_status(): raise SystemExit("⛔ مفتاح API غير صالح أو IP محظور.")
    while True:
        try:
            if not check_api_status():
                raise SystemExit("⛔ API محظور - توقف.")
            trade_logic()
            follow_trades()
            send_status()
        except Exception as e:
            send_telegram(f"❌ خطأ عام: {e}")
        time.sleep(300)
