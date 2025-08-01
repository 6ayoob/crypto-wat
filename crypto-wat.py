import asyncio
import logging
import aiohttp
from aiogram import Bot, Dispatcher
from config import TELEGRAM_TOKEN, ALLOWED_USER_IDS, BYBIT_API_KEY, BYBIT_API_SECRET
import time
import hmac
import hashlib
import urllib.parse

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

BYBIT_SPOT_API = "https://api.bybit.com"

# إعدادات التداول
SYMBOL = "BTCUSDT"   # العملة التي تريد تداولها تلقائياً
RISK_PERCENT = 10    # نسبة الرصيد المستخدم لكل صفقة (10%)
STOP_LOSS_PERCENT = 2  # وقف خسارة 2% تحت سعر الشراء
PROFIT_TARGET_PERCENT = 5  # هدف ربح 5%

# حالة الصفقة
current_position = None  # None أو dict مع معلومات الصفقة

# توقيع طلب Bybit
def generate_signature(secret, params):
    ordered_params = '&'.join([f"{k}={params[k]}" for k in sorted(params)])
    return hmac.new(secret.encode(), ordered_params.encode(), hashlib.sha256).hexdigest()

async def bybit_request(method, endpoint, params=None, private=False):
    params = params or {}
    if private:
        params.update({
            'api_key': BYBIT_API_KEY,
            'timestamp': int(time.time() * 1000),
            'recv_window': 5000,
        })
        signature = generate_signature(BYBIT_API_SECRET, params)
        params['sign'] = signature

    url = BYBIT_SPOT_API + endpoint
    if method.upper() == "GET":
        url += '?' + urllib.parse.urlencode(params)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.json()
    elif method.upper() == "POST":
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=params) as resp:
                return await resp.json()
    else:
        raise ValueError("Unsupported HTTP method")

async def get_spot_balance(coin="USDT"):
    res = await bybit_request("GET", "/spot/v1/account", private=True)
    if res.get("ret_code") == 0:
        balances = res.get("result", {}).get("balances", [])
        for b in balances:
            if b["coin"] == coin:
                return float(b["free"])
    return 0.0

async def get_price(symbol):
    res = await bybit_request("GET", "/spot/quote/v1/ticker/price", {"symbol": symbol})
    if res.get("ret_code") == 0:
        return float(res["result"]["price"])
    return 0.0

async def place_order(symbol, side, quantity):
    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": "MARKET",
        "qty": str(quantity),
        "time_in_force": "GTC",
    }
    res = await bybit_request("POST", "/spot/v1/order", params, private=True)
    if res.get("ret_code") == 0:
        return res["result"]
    return None

async def cancel_all_stop_orders(symbol):
    res = await bybit_request("GET", "/spot/v1/open-orders", {"symbol": symbol}, private=True)
    if res.get("ret_code") == 0:
        orders = res.get("result", {}).get("data", [])
        for o in orders:
            if o.get("type", "").startswith("STOP"):
                await bybit_request("POST", "/spot/v1/cancel-order", {
                    "symbol": symbol,
                    "order_id": o["order_id"]
                }, private=True)

async def place_stop_loss_limit(symbol, side, quantity, stop_price):
    # أمر وقف خسارة من نوع STOP_LIMIT
    params = {
        "symbol": symbol,
        "side": "SELL" if side == "BUY" else "BUY",
        "type": "STOP_LIMIT",
        "qty": str(quantity),
        "price": str(stop_price),
        "stop_px": str(stop_price),
        "time_in_force": "GTC",
    }
    res = await bybit_request("POST", "/spot/v1/order", params, private=True)
    if res.get("ret_code") == 0:
        return res["result"]
    return None

async def send_telegram(msg):
    for user_id in ALLOWED_USER_IDS:
        try:
            await bot.send_message(chat_id=user_id, text=msg)
        except Exception as e:
            logging.error(f"Failed to send message to {user_id}: {e}")

async def trading_loop():
    global current_position
    while True:
        price = await get_price(SYMBOL)
        balance = await get_spot_balance("USDT")
        coin = SYMBOL.replace("USDT", "")
        coin_balance = await get_spot_balance(coin)

        if current_position is None:
            # لم يتم فتح صفقة، اشترِ تلقائياً إذا توفرت شروطك (مثلاً السعر أقل من رقم معين أو إشارة)
            # هنا مثال: نشتري إذا السعر أقل من 30000 (عدل الشرط حسب استراتيجيتك)
            if price > 0 and price < 30000:
                qty = (balance * RISK_PERCENT / 100) / price
                qty = round(qty, 6)
                order = await place_order(SYMBOL, "BUY", qty)
                if order:
                    current_position = {
                        "side": "BUY",
                        "qty": qty,
                        "entry_price": price,
                    }
                    await send_telegram(f"✅ تم شراء {qty} {SYMBOL} بسعر {price} USDT")
                    # ضع وقف خسارة 2% تحت سعر الشراء
                    stop_price = round(price * (1 - STOP_LOSS_PERCENT / 100), 6)
                    await cancel_all_stop_orders(SYMBOL)
                    stop_order = await place_stop_loss_limit(SYMBOL, "BUY", qty, stop_price)
                    if stop_order:
                        await send_telegram(f"🛑 تم وضع وقف خسارة عند {stop_price} USDT")
        else:
            # لدينا صفقة مفتوحة، نراقب وقف الخسارة وهدف الربح
            entry = current_position["entry_price"]
            qty = current_position["qty"]
            side = current_position["side"]

            # تحديث وقف الخسارة لو ارتفع السعر (Trailing Stop)
            new_stop = round(price * (1 - STOP_LOSS_PERCENT / 100), 6)

            # إذا ارتفع وقف الخسارة (أي السعر ارتفع من دخول الصفقة)
            if new_stop > entry * (1 - STOP_LOSS_PERCENT / 100):
                await cancel_all_stop_orders(SYMBOL)
                stop_order = await place_stop_loss_limit(SYMBOL, side, qty, new_stop)
                if stop_order:
                    await send_telegram(f"🔄 تحديث وقف الخسارة إلى {new_stop} USDT")

            # تحقق هدف الربح: بيع إذا السعر زاد عن الهدف
            profit_target_price = round(entry * (1 + PROFIT_TARGET_PERCENT / 100), 6)
            if price >= profit_target_price:
                order = await place_order(SYMBOL, "SELL", qty)
                if order:
                    await send_telegram(f"🎉 تم بيع {qty} {SYMBOL} بسعر {price} USDT - هدف الربح تحقق!")
                    current_position = None
                    await cancel_all_stop_orders(SYMBOL)

            # تحقق وقف الخسارة: بيع إذا السعر هبط أقل من وقف الخسارة (يمكن تفعيل عبر webhook أو تحديثات إضافية)

        await asyncio.sleep(30)  # انتظر 30 ثانية ثم تحقق مرة أخرى

async def main():
    asyncio.create_task(trading_loop())
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())
