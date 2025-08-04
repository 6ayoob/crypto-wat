import json
import os
from okx_api import get_balance, get_last_price, place_limit_order, place_market_order
import requests

POSITIONS_FILE = "positions.json"
TRADING_AMOUNT = 10  # دولار لكل صفقة
STOP_LOSS_PERCENT = 3  # 3%
TAKE_PROFIT_PERCENT = 5  # 5%

# إعدادات Telegram
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

def enter_trade(symbol):
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

        # تحقق وقف خسارة
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

        # تحقق هدف ربح
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

    # إزالة الصفقات التي انتهت
    for sym in to_remove:
        positions.pop(sym, None)
    save_positions(positions)
