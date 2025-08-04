import json
import os
import requests
from okx_api import get_last_price, place_limit_order, place_market_order, get_historical_candles
import pandas as pd
import numpy as np
from datetime import datetime

POSITIONS_FILE = "positions.json"
TRADING_AMOUNT = 25  # دولار لكل صفقة
STOP_LOSS_PERCENT = 3
TAKE_PROFIT_PERCENT = 5

# Telegram إعدادات
TELEGRAM_TOKEN = "8300868885:AAEx8Zxdkz9CRUHmjJ0vvn6L3kC2kOPCHuk"
TELEGRAM_CHAT_ID = "658712542"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception as e:
        print(f"❌ Telegram Error: {e}")

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_positions(data):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_symbol(symbol):
    candles = get_historical_candles(symbol, bar="1D", limit=30)
    if not candles:
        print(f"❌ لا يمكن جلب الشموع لـ {symbol}")
        return False

    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "volCcy", "volCcyQuote", "confirm"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df["rsi"] = calculate_rsi(df["close"])

    avg_volume = df["volume"].rolling(window=14).mean().iloc[-1]
    current_volume = df["volume"].iloc[-1]

    today_close = df["close"].iloc[-1]
    yesterday_close = df["close"].iloc[-2]

    if df["rsi"].iloc[-1] < 30 and current_volume > avg_volume and today_close > yesterday_close:
        return True
    return False

def enter_trade(symbol):
    if not analyze_symbol(symbol):
        print(f"⚠️ {symbol} لا تحقق شروط الدخول.")
        return False

    price = get_last_price(symbol)
    if price is None:
        print(f"❌ لم أتمكن من جلب سعر {symbol}")
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

    save_positions(positions)
    for sym in to_remove:
        positions.pop(sym, None)
    save_positions(positions)

def is_market_bearish(symbols):
    down_count = 0
    total = len(symbols)
    for symbol in symbols:
        candles = get_historical_candles(symbol, bar="1D", limit=2)
        if not candles or len(candles) < 2:
            continue
        close_yesterday = float(candles[-2][4])
        close_today = float(candles[-1][4])
        if close_today < close_yesterday:
            down_count += 1
    if total == 0:
        return False
    percent_down = down_count / total * 100
    print(f"📉 نسبة العملات الهابطة اليوم: {percent_down:.2f}%")
    return percent_down >= 70

def generate_daily_report():
    positions = load_positions()
    lines = [f"📊 تقرير التداول اليومي - {datetime.now().strftime('%Y-%m-%d')}"]
    lines.append(f"عدد الصفقات المفتوحة: {len(positions)}")

    for symbol, pos in positions.items():
        current_price = get_last_price(symbol)
        if current_price:
            pnl = (current_price - pos["entry_price"]) * pos["size"]
            lines.append(f"{symbol}: الدخول عند {pos['entry_price']}, السعر الحالي {current_price}, الربح/الخسارة التقريبية: {pnl:.2f} USDT")
        else:
            lines.append(f"{symbol}: بيانات السعر غير متوفرة")

    report = "\n".join(lines)
    send_telegram_message(report)
