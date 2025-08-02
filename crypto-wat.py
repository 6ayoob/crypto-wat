import time
import requests
import hmac
import hashlib
import base64
import datetime
import json
from config import OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

BASE_URL = "https://www.okx.com"
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
    return {
        "Content-Type": "application/json",
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
        # لا تضع x-simulated-trading هنا لأنك تستخدم حساب حقيقي
    }

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

def place_order(instId, side, sz):
    path = "/api/v5/trade/order"
    body = {
        "instId": instId,
        "tdMode": "cash",
        "side": side.lower(),
        "ordType": "market",
        "sz": str(sz)
    }
    body_str = json.dumps(body)
    try:
        r = requests.post(BASE_URL + path, headers=okx_headers("POST", path, body_str), data=body_str, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        send_telegram(f"❌ خطأ في تنفيذ أمر {side} لـ {instId}: {e}")
        return None

def get_balance(ccy="USDT"):
    path = f"/api/v5/account/balance?ccy={ccy}"
    try:
        r = requests.get(BASE_URL + path, headers=okx_headers("GET", path), timeout=10)
        r.raise_for_status()
        balances = r.json().get("data", [])[0].get("details", [])
        for b in balances:
            if b["ccy"] == ccy:
                return float(b["availBal"])
    except Exception as e:
        send_telegram(f"❌ خطأ في جلب الرصيد: {e}")
    return 0.0

def trade_logic():
    global open_trades
    try:
        with open("coins.txt", "r") as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
    except FileNotFoundError:
        send_telegram("❌ ملف coins.txt غير موجود")
        return

    balance = get_balance()
    if balance < 30:
        send_telegram(f"⚠️ الرصيد غير كافي: {balance:.2f} USDT")
        return

    send_telegram(f"📊 بدء المسح لـ {len(symbols)} عملة. الرصيد الحالي: {balance:.2f} USDT")

    for s in symbols:
        instId = f"{s}-USDT"
        data = get_klines(instId)
        if not data or len(data) < 20:
            send_telegram(f"⚠️ بيانات غير كافية لـ {instId}، تم تخطيه")
            continue

        closes = [float(c[4]) for c in data]
        ema9 = calculate_ema(closes[-20:], 9)
        ema21 = calculate_ema(closes[-20:], 21)

        if ema9 and ema21 and ema9 > ema21 and not any(t["instId"] == instId for t in open_trades):
            price = closes[-1]
            qty = round(30 / price, 4)
            if place_order(instId, "buy", qty):
                open_trades.append({
                    "instId": instId, "symbol": s, "qty": qty, "entry_price": price,
                    "target1": price * 1.05, "target2": price * 1.10, "stop_loss": price * 0.98,
                    "sold_target1": False
                })
                send_telegram(f"✅ شراء {s} بسعر {price:.4f} | كمية: {qty}")
        time.sleep(1)

def follow_trades():
    global open_trades
    updated = []
    for t in open_trades:
        data = get_klines(t["instId"])
        if not data:
            updated.append(t)
            continue
        current_price = float(data[-1][4])
        if not t["sold_target1"] and current_price >= t["target1"]:
            half_qty = round(t["qty"] / 2, 4)
            if place_order(t["instId"], "sell", half_qty):
                send_telegram(f"🎯 بيع 50% من {t['symbol']} عند +5% | السعر: {current_price:.4f}")
                t["sold_target1"] = True
                t["qty"] -= half_qty
            updated.append(t)
        elif current_price >= t["target2"]:
            if place_order(t["instId"], "sell", t["qty"]):
                send_telegram(f"🏁 بيع باقي {t['symbol']} عند +10% | السعر: {current_price:.4f}")
        elif current_price <= t["stop_loss"]:
            if place_order(t["instId"], "sell", t["qty"]):
                send_telegram(f"🚨 وقف خسارة {t['symbol']} | السعر: {current_price:.4f}")
        else:
            updated.append(t)
    open_trades = updated

if __name__ == "__main__":
    try:
        ip = requests.get("https://api.ipify.org?format=json", timeout=5).json().get("ip", "غير معروف")
        send_telegram(f"🤖 بدأ تشغيل البوت على IP: {ip}")
    except:
        send_telegram("🤖 بدأ تشغيل البوت (تعذر جلب IP)")

    while True:
        try:
            trade_logic()
            follow_trades()
        except Exception as e:
            send_telegram(f"❌ خطأ عام: {e}")
        time.sleep(300)
