import time
import requests
import hmac
import hashlib
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API, BINANCE_SECRET

open_trades = []
last_report_time = 0  # ⬅️ لتقرير دوري كل ساعة

# ✅ إرسال رسالة Telegram
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except:
        print("⚠️ فشل إرسال تنبيه إلى Telegram")

# ✅ جلب IP الخارجي
def get_external_ip():
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=5)
        return r.json().get("ip", "غير معروف")
    except:
        return "غير معروف"

# ✅ جلب توقيت Binance
def get_binance_server_time():
    try:
        r = requests.get("https://api.binance.com/api/v3/time", timeout=5)
        r.raise_for_status()
        return r.json()['serverTime']
    except Exception as e:
        send_telegram(f"🚨 لا يمكن جلب توقيت Binance: {e}")
        raise SystemExit("⛔ توقيت Binance غير متاح، إيقاف البوت لحماية الحساب.")

# ✅ إنشاء توقيع HMAC
def sign_params(params):
    query = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query + "&signature=" + signature

# ✅ فحص حالة API Key (مكرر دوريًا)
def check_api_status():
    url = "https://api.binance.com/api/v3/account"
    params = {"timestamp": get_binance_server_time(), "recvWindow": 10000}
    headers = {"X-MBX-APIKEY": BINANCE_API}
    query = sign_params(params)
    full_url = f"{url}?{query}"
    try:
        r = requests.get(full_url, headers=headers, timeout=5)
        if r.status_code == 200:
            return True
        elif r.status_code == 451:
            send_telegram("❌ Binance رفض المفتاح (451) - تحتاج لتفعيل IP Whitelist.")
        elif r.status_code == 401:
            send_telegram("❌ مفتاح API غير صالح (401) - تحقق من API_KEY و SECRET.")
        else:
            send_telegram(f"⚠️ خطأ API: {r.status_code} - {r.text}")
        return False
    except Exception as e:
        send_telegram(f"🚨 فشل الاتصال بـ Binance: {e}")
        return False

# ✅ جلب الرصيد مع معالجة الأخطاء
def get_balance(asset="USDT"):
    url = "https://api.binance.com/api/v3/account"
    params = {"timestamp": get_binance_server_time(), "recvWindow": 10000}
    headers = {"X-MBX-APIKEY": BINANCE_API}
    query = sign_params(params)
    try:
        r = requests.get(f"{url}?{query}", headers=headers, timeout=5)
        if r.status_code == 451:
            send_telegram("❌ 451: IP غير مسموح. أضف IP للسيرفر في Binance.")
            return 0.0
        r.raise_for_status()
        balances = r.json().get("balances", [])
        return float(next((b["free"] for b in balances if b["asset"] == asset), 0.0))
    except Exception as e:
        send_telegram(f"❌ خطأ جلب الرصيد: {e}")
        return 0.0

# ✅ جلب بيانات الأسعار
def get_klines(symbol):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": symbol, "interval": "15m", "limit": 50}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        send_telegram(f"❌ خطأ جلب بيانات {symbol}: {e}")
        return []

# ✅ حساب EMA
def calculate_ema(prices, length):
    if len(prices) < length: return None
    k, ema = 2 / (length + 1), prices[0]
    for p in prices[1:]: ema = p * k + ema * (1 - k)
    return ema

# ✅ تنفيذ أوامر شراء/بيع
def place_order(symbol, side, qty):
    url = "https://api.binance.com/api/v3/order"
    params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty, "timestamp": get_binance_server_time(), "recvWindow": 10000}
    headers = {"X-MBX-APIKEY": BINANCE_API}
    query = sign_params(params)
    try:
        r = requests.post(f"{url}?{query}", headers=headers, timeout=5)
        if r.status_code == 451:
            send_telegram(f"❌ أمر {side} مرفوض (451) - IP غير مسموح.")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        send_telegram(f"❌ خطأ تنفيذ {side} لـ {symbol}: {e}")
        return None

# ✅ منطق التداول (كما هو مع فحص الرصيد والـ EMA)
def trade_logic():
    global open_trades
    try:
        with open("coins.txt") as f:
            symbols = [x.strip().upper() for x in f if x.strip()]
    except:
        send_telegram("❌ ملف coins.txt غير موجود")
        return

    usdt = get_balance()
    if usdt < 30:
        send_telegram(f"⚠️ الرصيد غير كافي: {usdt:.2f} USDT")
        return

    send_telegram(f"📊 فحص السوق لـ {len(symbols)} عملة - الرصيد: {usdt:.2f} USDT")

    for s in symbols:
        pair = s + "USDT"
        data = get_klines(pair)
        if len(data) < 20: continue

        closes = [float(k[4]) for k in data]
        ema9, ema21 = calculate_ema(closes[-20:], 9), calculate_ema(closes[-20:], 21)
        if not ema9 or not ema21: continue

        if ema9 > ema21 and not any(t["symbol_pair"] == pair for t in open_trades):
            price = closes[-1]; qty = round(30 / price, 5)
            order = place_order(pair, "BUY", qty)
            if order:
                send_telegram(f"✅ شراء {s} بسعر {price:.4f}, كمية {qty}")
                open_trades.append({"symbol_pair": pair, "symbol": s, "qty": qty, "entry_price": price,
                                    "target1": price * 1.05, "target2": price * 1.10, "stop_loss": price * 0.98,
                                    "sold_target1": False})
        time.sleep(1)

# ✅ متابعة الصفقات المفتوحة
def follow_trades():
    global open_trades
    still_open = []
    for t in open_trades:
        pair = t["symbol_pair"]
        data = get_klines(pair)
        if not data: continue
        price = float(data[-1][4])

        if not t["sold_target1"] and price >= t["target1"]:
            half = round(t["qty"] / 2, 5)
            place_order(pair, "SELL", half)
            send_telegram(f"🎯 بيع نصف {t['symbol']} عند +5% بسعر {price:.4f}")
            t["sold_target1"] = True; t["qty"] -= half
        elif price >= t["target2"]:
            place_order(pair, "SELL", t["qty"])
            send_telegram(f"🏁 بيع كامل {t['symbol']} عند +10% بسعر {price:.4f}")
        elif price <= t["stop_loss"]:
            place_order(pair, "SELL", t["qty"])
            send_telegram(f"🚨 وقف خسارة {t['symbol']} عند {price:.4f}")
        else:
            still_open.append(t)

    open_trades = still_open

# ✅ تقرير دوري
def send_status_report():
    global last_report_time
    if time.time() - last_report_time > 3600:  # كل ساعة
        ip = get_external_ip()
        send_telegram(f"📡 تقرير دوري:\n- IP: {ip}\n- صفقات مفتوحة: {len(open_trades)}")
        last_report_time = time.time()

# ✅ التشغيل الرئيسي
if __name__ == "__main__":
    ip = get_external_ip()
    send_telegram(f"🤖 بدء تشغيل البوت - IP: {ip}")
    if not check_api_status(): raise SystemExit("⛔ المفتاح غير صالح - توقف البوت.")
    
    while True:
        try:
            if not check_api_status():  # إعادة تحقق دوري
                raise SystemExit("⛔ المفتاح مرفوض - توقف البوت.")
            trade_logic()
            follow_trades()
            send_status_report()
        except Exception as e:
            send_telegram(f"❌ خطأ عام: {e}")
        time.sleep(300)
