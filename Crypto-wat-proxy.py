import time
import requests
import hmac
import hashlib
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API, BINANCE_SECRET

# ✅ رابط Proxy الخاص بك (Cloudflare Worker)
PROXY_URL = "https://long-flower-6e9b.tayoob632.workers.dev"

open_trades = []

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print(f"Telegram Error: {e}")

def get_binance_server_time():
    try:
        response = requests.get(f"{PROXY_URL}/api/v3/time", timeout=10)
        response.raise_for_status()
        return response.json()['serverTime']
    except Exception as e:
        send_telegram(f"⛔ فشل جلب توقيت Binance عبر Proxy: {e}")
        return None

def sign_request(params):
    query = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    return query, signature

def get_klines(symbol):
    try:
        url = f"{PROXY_URL}/api/v3/klines?symbol={symbol}&interval=15m&limit=50"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        send_telegram(f"❌ خطأ جلب بيانات {symbol}: {e}")
        return []

def calculate_ema(prices, length):
    if len(prices) < length: return None
    k = 2 / (length + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema

def place_order(symbol, side, qty):
    ts = get_binance_server_time()
    if ts is None: return None
    params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": qty, "timestamp": ts, "recvWindow": 10000}
    query, sig = sign_request(params)
    url = f"{PROXY_URL}/api/v3/order?{query}&signature={sig}"
    try:
        r = requests.post(url, headers={"X-MBX-APIKEY": BINANCE_API}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        send_telegram(f"❌ خطأ تنفيذ أمر {side} {symbol}: {e}")
        return None

def get_balance(asset="USDT"):
    ts = get_binance_server_time()
    if ts is None: return 0.0
    params = {"timestamp": ts, "recvWindow": 10000}
    query, sig = sign_request(params)
    url = f"{PROXY_URL}/api/v3/account?{query}&signature={sig}"
    try:
        r = requests.get(url, headers={"X-MBX-APIKEY": BINANCE_API}, timeout=10)
        r.raise_for_status()
        for b in r.json().get("balances", []):
            if b["asset"] == asset:
                return float(b["free"])
    except Exception as e:
        send_telegram(f"❌ خطأ جلب الرصيد: {e}")
    return 0.0

def trade_logic():
    global open_trades
    try:
        with open("coins.txt", "r") as f:
            symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        send_telegram("❌ ملف coins.txt غير موجود")
        return
    
    balance = get_balance()
    if balance < 30:
        send_telegram(f"⚠️ الرصيد غير كافي: {balance:.2f} USDT")
        return
    
    send_telegram(f"📈 بدء الفحص لـ {len(symbols)} عملة. الرصيد: {balance:.2f} USDT")

    for s in symbols:
        pair = s + "USDT"
        data = get_klines(pair)
        if not data: continue
        closes = [float(c[4]) for c in data]
        ema9 = calculate_ema(closes[-20:], 9)
        ema21 = calculate_ema(closes[-20:], 21)
        if ema9 and ema21 and ema9 > ema21 and not any(t["symbol_pair"] == pair for t in open_trades):
            price = closes[-1]
            qty = round(30 / price, 5)
            if place_order(pair, "BUY", qty):
                open_trades.append({"symbol_pair": pair, "symbol": s, "qty": qty, "entry_price": price,
                                    "target1": price*1.05, "target2": price*1.10, "stop_loss": price*0.98,
                                    "sold_target1": False})
                send_telegram(f"✅ شراء {s} بسعر {price:.4f} كمية: {qty}")

def follow_trades():
    global open_trades
    updated = []
    for t in open_trades:
        data = get_klines(t["symbol_pair"])
        if not data: continue
        price = float(data[-1][4])
        if not t["sold_target1"] and price >= t["target1"]:
            half = round(t["qty"]/2,5)
            place_order(t["symbol_pair"],"SELL",half)
            t["sold_target1"]=True
            t["qty"]-=half
            send_telegram(f"🎯 بيع نصف {t['symbol']} عند +5% بسعر {price:.4f}")
        elif price >= t["target2"]:
            place_order(t["symbol_pair"],"SELL",t["qty"])
            send_telegram(f"🏁 بيع كامل {t['symbol']} عند +10% بسعر {price:.4f}")
        elif price <= t["stop_loss"]:
            place_order(t["symbol_pair"],"SELL",t["qty"])
            send_telegram(f"🚨 وقف خسارة {t['symbol']} عند {price:.4f}")
        else:
            updated.append(t)
    open_trades = updated

if __name__ == "__main__":
    send_telegram("🤖 بدأ تشغيل البوت باستخدام Cloudflare Proxy")
    while True:
        trade_logic()
        follow_trades()
        time.sleep(300)
