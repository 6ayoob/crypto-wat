import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from okx_api import get_last_price, place_limit_order, place_market_order, get_historical_candles
from config import TRADING_AMOUNT, STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

POSITIONS_FILE = "positions.json"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_positions(data):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def analyze_symbol(symbol):
    candles = get_historical_candles(symbol, '1h', 60)
    if not candles or len(candles) < 50:
        return False

    df = pd.DataFrame(candles, columns=['timestamp','open','high','low','close','volume'])
    df['close'] = pd.to_numeric(df['close'])
    df['MA_fast'] = df['close'].rolling(window=20).mean()
    df['MA_slow'] = df['close'].rolling(window=50).mean()

    if pd.isna(df['MA_fast'].iloc[-1]) or pd.isna(df['MA_slow'].iloc[-1]):
        return False

    ma_fast_current = df['MA_fast'].iloc[-1]
    ma_slow_current = df['MA_slow'].iloc[-1]
    ma_fast_prev = df['MA_fast'].iloc[-2]
    ma_slow_prev = df['MA_slow'].iloc[-2]

    # إشارة شراء: تقاطع الماكد السريع فوق البطيء
    if ma_fast_prev <= ma_slow_prev and ma_fast_current > ma_slow_current:
        return True
    return False

def enter_trade(symbol):
    if not analyze_symbol(symbol):
        print(f"{symbol}: لا تحقق شروط الدخول")
        return False

    price = get_last_price(symbol)
    if price is None:
        print(f"{symbol}: فشل في جلب السعر الحالي")
        return False

    size = round(TRADING_AMOUNT / price, 6)
    print(f"دخول صفقة شراء لـ {symbol} بسعر {price} وحجم {size}")

    order = place_limit_order(symbol, "buy", price, size)
    if order:
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
        send_telegram_message(f"❌ فشل شراء {symbol}")
        return False

def check_positions():
    positions = load_positions()
    if not positions:
        print("لا توجد صفقات مفتوحة حالياً.")
        return

    to_remove = []
    for symbol, pos in positions.items():
        current_price = get_last_price(symbol)
        if current_price is None:
            continue

        # تحديث وقف الخسارة (يمكن تعديل هذه الاستراتيجية حسب رغبتك)
        new_stop_loss = max(pos["stop_loss"], round(current_price * (1 - STOP_LOSS_PERCENT / 100), 6))
        if new_stop_loss > pos["stop_loss"]:
            pos["stop_loss"] = new_stop_loss

        if current_price <= pos["stop_loss"]:
            print(f"وقف خسارة مفعل على {symbol} عند السعر {current_price}")
            sell_result = place_market_order(symbol, "sell", pos["size"])
            if sell_result:
                send_telegram_message(f"⚠️ تم بيع {symbol} عند وقف الخسارة بسعر {current_price}")
                to_remove.append(symbol)

        elif current_price >= pos["take_profit"]:
            print(f"تم الوصول لهدف الربح على {symbol} عند السعر {current_price}")
            sell_result = place_market_order(symbol, "sell", pos["size"])
            if sell_result:
                send_telegram_message(f"🎯 تم بيع {symbol} عند هدف الربح بسعر {current_price}")
                to_remove.append(symbol)

    for sym in to_remove:
        positions.pop(sym, None)
    save_positions(positions)

def get_positions_summary():
    positions = load_positions()
    if not positions:
        return "لا توجد صفقات مفتوحة حالياً."
    text = "📋 صفقات مفتوحة:\n"
    for symbol, pos in positions.items():
        current_price = get_last_price(symbol) or 0
        pnl = (current_price - pos["entry_price"]) * pos["size"]
        text += f"{symbol} | سعر الدخول: {pos['entry_price']:.4f} | السعر الحالي: {current_price:.4f} | ربح/خسارة: {pnl:.2f} USDT\n"
    return text
