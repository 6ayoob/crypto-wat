import json
import os
from okx_api import get_balance, get_last_price, place_limit_order, place_market_order

POSITIONS_FILE = "positions.json"
TRADING_AMOUNT = 10  # دولار لكل صفقة
STOP_LOSS_PERCENT = 3  # وقف خسارة ثابت أولي %
TAKE_PROFIT_PERCENT = 5  # هدف ربح %
TRAILING_START_PERCENT = 1  # بعد الربح 1% يبدأ وقف الخسارة بالتحرك
TRAILING_STEP_PERCENT = 0.5  # وقف الخسارة يتحرك ليصبح أقل من السعر الأعلى بـ 0.5%

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
        positions = load_positions()
        positions[symbol] = {
            "entry_price": price,
            "size": size,
            "stop_loss": round(price * (1 - STOP_LOSS_PERCENT / 100), 6),
            "take_profit": round(price * (1 + TAKE_PROFIT_PERCENT / 100), 6),
            "highest_price": price  # لتتبع أعلى سعر للوقف المتحرك
        }
        save_positions(positions)
        return True
    else:
        print(f"❌ فشل في دخول الصفقة: {result}")
        return False

def place_market_order(instId, side, size):
    # تنفيذ أمر سوقي Market order
    from okx_api import okx_request
    body = {
        "instId": instId,
        "tdMode": "cash",
        "side": side,
        "ordType": "market",
        "sz": str(size)
    }
    return okx_request("POST", "/api/v5/trade/order", data=body)

def check_positions():
    positions = load_positions()
    to_remove = []
    for symbol, pos in positions.items():
        current_price = get_last_price(symbol)
        if current_price is None:
            continue

        # تحديث أعلى سعر وصل له
        if current_price > pos["highest_price"]:
            positions[symbol]["highest_price"] = current_price

        # حساب وقف خسارة متحرك
        trailing_stop = round(
            positions[symbol]["highest_price"] * (1 - TRAILING_STEP_PERCENT / 100), 6
        )

        # إذا الربح الحالي تجاوز نقطة البداية للتحرك
        profit_percent = (current_price - pos["entry_price"]) / pos["entry_price"] * 100

        if profit_percent > TRAILING_START_PERCENT and trailing_stop > pos["stop_loss"]:
            positions[symbol]["stop_loss"] = trailing_stop

        # تحقق وقف خسارة (بما فيه المتحرك)
        if current_price <= positions[symbol]["stop_loss"]:
            print(f"⚠️ وقف خسارة مفعل على {symbol} عند السعر {current_price}")
            res = place_market_order(symbol, "sell", pos["size"])
            if res and res.get("code") == "0":
                print(f"✅ بيع {symbol} بنجاح عند وقف الخسارة")
                to_remove.append(symbol)
            else:
                print(f"❌ فشل بيع {symbol} عند وقف الخسارة: {res}")

        # تحقق هدف ربح
        elif current_price >= pos["take_profit"]:
            print(f"🎯 تم الوصول لهدف ربح على {symbol} عند السعر {current_price}")
            res = place_market_order(symbol, "sell", pos["size"])
            if res and res.get("code") == "0":
                print(f"✅ بيع {symbol} بنجاح عند هدف الربح")
                to_remove.append(symbol)
            else:
                print(f"❌ فشل بيع {symbol} عند هدف الربح: {res}")

    # إزالة الصفقات التي انتهت
    for sym in to_remove:
        positions.pop(sym, None)
    save_positions(positions)
