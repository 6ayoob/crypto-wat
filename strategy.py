import json
import os
import requests
from okx_api import get_last_price, place_limit_order, place_market_order, get_historical_candles
import pandas as pd
from datetime import datetime

POSITIONS_FILE = "positions.json"
TRADING_AMOUNT = 15  # 15 دولار لكل صفقة
STOP_LOSS_PERCENT = 3
TAKE_PROFIT_PERCENT = 4

TELEGRAM_TOKEN = "8300868885:AAEx8Zxdkz9CRUHmjJ0vvn6L3kC2kOPCHuk"
TELEGRAM_CHAT_ID = "658712542"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        if not response.ok:
            print(f"❌ خطأ في إرسال رسالة تيليجرام: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Telegram Exception: {e}")

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ ملف المراكز معطوب، سيتم إنشاء ملف جديد.")
            return {}
    return {}

def save_positions(data):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def analyze_symbol(symbol):
    candles = get_historical_candles(symbol, bar="1H", limit=100)
    if not candles:
        print(f"❌ لا يمكن جلب الشموع لـ {symbol}")
        return False

    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "volCcy", "volCcyQuote", "confirm"
    ])
    df["close"] = pd.to_numeric(df["close"])

    df["MA_fast"] = df["close"].rolling(window=20).mean()
    df["MA_slow"] = df["close"].rolling(window=50).mean()

    if len(df) < 51 or pd.isna(df["MA_fast"].iloc[-1]) or pd.isna(df["MA_slow"].iloc[-1]):
        return False

    ma_fast_current = df["MA_fast"].iloc[-1]
    ma_slow_current = df["MA_slow"].iloc[-1]
    ma_fast_prev = df["MA_fast"].iloc[-2]
    ma_slow_prev = df["MA_slow"].iloc[-2]

    if ma_fast_prev <= ma_slow_prev and ma_fast_current > ma_slow_current:
        return True

    return False

def enter_trade(symbol):
    if not analyze_symbol(symbol):
        print(f"⚠️ {symbol} لا تحقق شروط الدخول.")
        return False

    price = get_last_price(symbol)
    if price is None or price > 20:
        print(f"❌ {symbol} سعره غير مناسب أو غير متوفر.")
        return False

    size = round(TRADING_AMOUNT / price, 6)
    print(f"➡️ محاولة دخول صفقة {symbol} بسعر {price} وحجم {size}")

    result = place_limit_order(symbol, "buy", price, size)
    if result and result.get("code") == "0":
        print(f"✅ أمر شراء {symbol} تم بنجاح!")
        send_telegram_message(f"✅ تم شراء {symbol} بسعر {price} وحجم {size}")
        positions = load_positions()
        positions[symbol] = {
            "entry_price": price,
            "size": size,
            "stop_loss": round(price * (1 - STOP_LOSS_PERCENT / 100), 6),
            "take_profit": round(price * (1 + TAKE_PROFIT_PERCENT / 100), 6)
        }
        save_positions(positions)
        return True
    else:
        print(f"❌ فشل في دخول الصفقة: {result}")
        send_telegram_message(f"❌ فشل شراء {symbol}: {result}")
        return False

def check_positions():
    positions = load_positions()
    if not positions:
        print("⚠️ لا توجد صفقات مفتوحة حالياً.")
        return

    to_remove = []
    for symbol, pos in positions.items():
        current_price = get_last_price(symbol)
        if current_price is None:
            continue

        new_stop_loss = max(pos["stop_loss"], round(current_price * (1 - STOP_LOSS_PERCENT / 100), 6))
        if new_stop_loss > pos["stop_loss"]:
            pos["stop_loss"] = new_stop_loss

        if current_price <= pos["stop_loss"]:
            print(f"⚠️ وقف خسارة مفعل على {symbol} عند السعر {current_price}")
            sell_result = place_market_order(symbol, "sell", pos["size"])
            if sell_result and sell_result.get("code") == "0":
                print(f"✅ تم بيع {symbol} عند وقف الخسارة بسعر السوق")
                send_telegram_message(f"⚠️ تم بيع {symbol} عند وقف الخسارة بسعر السوق: {current_price}")
                to_remove.append(symbol)
            else:
                print(f"❌ فشل بيع {symbol} عند وقف الخسارة: {sell_result}")
                send_telegram_message(f"❌ فشل بيع {symbol} عند وقف الخسارة: {sell_result}")

        elif current_price >= pos["take_profit"]:
            print(f"🎯 تم الوصول لهدف ربح على {symbol} عند السعر {current_price}")
            sell_result = place_market_order(symbol, "sell", pos["size"])
            if sell_result and sell_result.get("code") == "0":
                print(f"✅ تم بيع {symbol} عند هدف الربح بسعر السوق")
                send_telegram_message(f"🎯 تم بيع {symbol} عند هدف الربح بسعر السوق: {current_price}")
                to_remove.append(symbol)
            else:
                print(f"❌ فشل بيع {symbol} عند هدف الربح: {sell_result}")
                send_telegram_message(f"❌ فشل بيع {symbol} عند هدف الربح: {sell_result}")

    for sym in to_remove:
        positions.pop(sym, None)
    save_positions(positions)
