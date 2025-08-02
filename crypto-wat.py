import time
import hmac
import hashlib
import base64
import datetime
import json
import requests
from config import OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

BASE_URL = "https://www.okx.com"

# قائمة العملات الثابتة (يمكنك تعديلها)
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "AVAX-USDT", "XRP-USDT",
    "LTC-USDT", "BCH-USDT", "DOT-USDT", "MATIC-USDT", "ATOM-USDT",
    "LINK-USDT", "ALGO-USDT", "VET-USDT", "EOS-USDT", "FIL-USDT",
    "TRX-USDT", "XLM-USDT", "NEAR-USDT", "AAVE-USDT", "UNI-USDT",
    "CAKE-USDT", "FTM-USDT", "GRT-USDT", "SUSHI-USDT", "YFI-USDT",
    "COMP-USDT", "MKR-USDT", "ZRX-USDT", "SNX-USDT", "BAT-USDT",
    "CRV-USDT", "ENJ-USDT", "1INCH-USDT", "KNC-USDT", "REN-USDT",
    "BAL-USDT", "NKN-USDT", "STORJ-USDT", "CHZ-USDT", "RUNE-USDT",
    "DASH-USDT", "ZEN-USDT", "ZEC-USDT", "OMG-USDT", "LRC-USDT",
    "CEL-USDT", "KSM-USDT", "BTG-USDT", "QTUM-USDT", "WAVES-USDT"
]

open_trades = []

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception as e:
        print(f"Telegram Error: {e}")

def iso_timestamp():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'

def sign_okx(ts, method, path, body=""):
    pre_hash = f"{ts}{method.upper()}{path}{body}"
    hmac_key = hmac.new(OKX_SECRET_KEY.encode(), pre_hash.encode(), hashlib.sha256)
    return base64.b64encode(hmac_key.digest()).decode()

def okx_headers(method, path, body=""):
    ts = iso_timestamp()
    sign = sign_okx(ts, method, path, body)
    headers = {
        "Content-Type": "application/json",
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
    }
    return headers

def get_klines(instId):
    url = f"/api/v5/market/candles?instId={instId}&bar=15m&limit=50"
    try:
        r = requests.get(BASE_URL + url, timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])[::-1]
    except Exception as e:
        send_telegram(f"❌ خطأ في جلب الشموع لـ {instId}: {e}")
        return []

def calculate_ema(prices, length):
    if len(prices) < length:
        return None
    k = 2 / (length + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period+1):
        delta = prices[i] - prices[i-1]
        if delta > 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(delta))
    average_gain = sum(gains) / period
    average_loss = sum(losses) / period
    if average_loss == 0:
        return 100
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_balance(ccy="USDT"):
    path = f"/api/v5/account/balance?ccy={ccy}"
    ts = iso_timestamp()
    sign = sign_okx(ts, "GET", path, "")
    headers = {
        "Content-Type": "application/json",
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
    }
    try:
        r = requests.get(BASE_URL + path, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        for item in data.get("data", []):
            details = item.get("details", [])
            for d in details:
                if d.get("ccy") == ccy:
                    return float(d.get("availBal", "0"))
    except Exception as e:
        send_telegram(f"❌ خطأ في جلب الرصيد: {e}")
    return 0.0

def place_order(instId, side, sz):
    path = "/api/v5/trade/order"
    body = {
        "instId": instId,
        "tdMode": "cash",
        "side": side.lower(),
        "ordType": "market",
        "sz": str(sz)
    }
    ts = iso_timestamp()
    sign = sign_okx(ts, "POST", path, json.dumps(body))
    headers = {
        "Content-Type": "application/json",
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
    }
    try:
        r = requests.post(BASE_URL + path, headers=headers, data=json.dumps(body), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        send_telegram(f"❌ خطأ تنفيذ أمر {side} لـ {instId}: {e}")
        return None

def trade_logic():
    global open_trades
    balance = get_balance()
    if balance < 30:
        send_telegram(f"⚠️ رصيد USDT غير كافي: {balance:.2f}")
        return

    # نحدد حجم الصفقة: 2% من الرصيد
   trade_size_usdt = balance * 0.30

    for instId in SYMBOLS:
        if any(t["instId"] == instId for t in open_trades):
            continue  # هناك صفقة مفتوحة

        data = get_klines(instId)
        if not data or len(data) < 21:
            send_telegram(f"⚠️ بيانات غير كافية لـ {instId}")
            continue

        closes = [float(candle[4]) for candle in data]
        ema9 = calculate_ema(closes[-20:], 9)
        ema21 = calculate_ema(closes[-20:], 21)
        rsi = calculate_rsi(closes[-20:])

        if ema9 is None or ema21 is None or rsi is None:
            continue

        # شرط الدخول
        if ema9 > ema21 and rsi < 70:
            qty = round(trade_size_usdt / closes[-1], 6)
            order = place_order(instId, "buy", qty)
            if order:
                open_trades.append({
                    "instId": instId,
                    "entry_price": closes[-1],
                    "qty": qty,
                    "sold": False
                })
                send_telegram(f"✅ شراء {instId} بسعر {closes[-1]:.4f} كمية: {qty}")
        time.sleep(1)  # احترام حدود API

def follow_trades():
    global open_trades
    updated_trades = []
    for trade in open_trades:
        data = get_klines(trade["instId"])
        if not data or len(data) == 0:
            updated_trades.append(trade)
            continue
        current_price = float(data[-1][4])
        entry_price = trade["entry_price"]
        qty = trade["qty"]

        # حساب الربح/الخسارة بالنسبة المئوية
        change_pct = (current_price - entry_price) / entry_price * 100

        # حساب EMA9 و EMA21 لتحديد تقاطع خروج
        closes = [float(candle[4]) for candle in data]
        ema9 = calculate_ema(closes[-20:], 9)
        ema21 = calculate_ema(closes[-20:], 21)

        # شروط الخروج
        if ema9 is not None and ema21 is not None:
            if ema9 < ema21 or change_pct <= -2 or change_pct >= 5:
                order = place_order(trade["instId"], "sell", qty)
                if order:
                    send_telegram(f"🏁 بيع {trade['instId']} بسعر {current_price:.4f} تغير %{change_pct:.2f}")
            else:
                updated_trades.append(trade)
        else:
            updated_trades.append(trade)

        time.sleep(1)

    open_trades = updated_trades

if __name__ == "__main__":
    send_telegram("🤖 بدأ تشغيل بوت التداول على OKX")

    while True:
        try:
            trade_logic()
            follow_trades()
        except Exception as e:
            send_telegram(f"❌ خطأ عام: {e}")
        time.sleep(300)  # كل 5 دقائق
