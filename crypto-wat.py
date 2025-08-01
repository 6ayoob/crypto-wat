import time
import requests
import hmac
import hashlib
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, BINANCE_API, BINANCE_SECRET

open_trades = []

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram send error: {e}")

def get_klines(symbol):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "15m", "limit": 50}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        send_telegram(f"❌ خطأ في جلب بيانات {symbol}: {e}")
        return []

def calculate_ema(prices, length):
    if len(prices) < length:
        return None
    k = 2 / (length + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = price * k + ema * (1 - k)
    return ema

def place_order(symbol, side, quantity):
timestamp = get_binance_server_time()
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": timestamp
    }
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(BINANCE_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_API}
    url = f"https://api.binance.com/api/v3/order?{query_string}&signature={signature}"
    try:
        response = requests.post(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        send_telegram(f"❌ خطأ في تنفيذ أمر {side} لـ {symbol}: {e}")
        return None

def get_balance(asset="USDT"):
    url = "https://api.binance.com/api/v3/account"
timestamp = get_binance_server_time()
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(BINANCE_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": BINANCE_API}
    url = f"{url}?{query_string}&signature={signature}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        balances = response.json().get("balances", [])
        for b in balances:
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0
    except Exception as e:
        send_telegram(f"❌ خطأ في جلب الرصيد: {e}")
        return 0.0

def trade_logic():
    global open_trades
    try:
        with open("coins.txt", "r") as f:
            symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        send_telegram("❌ ملف coins.txt غير موجود")
        return

    usdt_balance = get_balance()
    if usdt_balance < 30:
        send_telegram(f"⚠️ الرصيد غير كافي للتداول: {usdt_balance:.2f} USDT")
        return

    send_telegram(f"📈 فحص السوق بدأ لـ {len(symbols)} عملة، الرصيد: {usdt_balance:.2f} USDT")

    for symbol in symbols:
        symbol_pair = symbol + "USDT"
        klines = get_klines(symbol_pair)

        if not klines or len(klines) < 20:
            send_telegram(f"⚠️ بيانات غير كافية لـ {symbol_pair}, تم تخطيه")
            time.sleep(1)
            continue

        try:
            closes = [float(k[4]) for k in klines]
        except Exception as e:
            send_telegram(f"❌ خطأ في معالجة بيانات الأسعار لـ {symbol_pair}: {e}")
            time.sleep(1)
            continue

        ema9 = calculate_ema(closes[-20:], 9)
        ema21 = calculate_ema(closes[-20:], 21)

        if ema9 is None or ema21 is None:
            send_telegram(f"⚠️ لا يمكن حساب EMA لـ {symbol_pair} بسبب نقص البيانات")
            time.sleep(1)
            continue

        # شراء عند اختراق EMA9 فوق EMA21 إذا لا توجد صفقة مفتوحة مسبقاً لهذا الرمز
        if ema9 > ema21 and not any(t["symbol_pair"] == symbol_pair for t in open_trades):
            price = closes[-1]
            qty = round(30 / price, 5)
            order = place_order(symbol_pair, "BUY", qty)
            if order:
                send_telegram(f"✅ تم شراء {symbol} بسعر {price:.4f}, كمية: {qty}")
                open_trades.append({
                    "symbol_pair": symbol_pair,
                    "symbol": symbol,
                    "qty": qty,
                    "entry_price": price,
                    "target1": price * 1.05,
                    "target2": price * 1.10,
                    "stop_loss": price * 0.98,
                    "sold_target1": False
                })

        time.sleep(1)  # احترام حدود API

def follow_trades():
    global open_trades
    still_open = []
    for trade in open_trades:
        symbol_pair = trade["symbol_pair"]
        qty = trade["qty"]
        try:
            klines = get_klines(symbol_pair)
            if not klines:
                continue
            current_price = float(klines[-1][4])

            if not trade["sold_target1"] and current_price >= trade["target1"]:
                half_qty = round(qty / 2, 5)
                place_order(symbol_pair, "SELL", half_qty)
                send_telegram(f"🎯 بيع 50% من {trade['symbol']} عند +5% بسعر {current_price:.4f}")
                trade["sold_target1"] = True
                trade["qty"] -= half_qty

            elif current_price >= trade["target2"]:
                place_order(symbol_pair, "SELL", trade["qty"])
                send_telegram(f"🏁 بيع الباقي من {trade['symbol']} عند +10% بسعر {current_price:.4f}")

            elif current_price <= trade["stop_loss"]:
                place_order(symbol_pair, "SELL", trade["qty"])
                send_telegram(f"🚨 وقف خسارة {trade['symbol']} عند {current_price:.4f}")

            else:
                still_open.append(trade)

        except Exception as e:
            send_telegram(f"❌ خطأ في متابعة الصفقة {trade['symbol']}: {e}")
            still_open.append(trade)
    open_trades = still_open

if __name__ == "__main__":
    while True:
        try:
            trade_logic()
            follow_trades()
        except Exception as e:
            send_telegram(f"❌ خطأ عام: {e}")
        time.sleep(300)  # كل 5 دقائق
