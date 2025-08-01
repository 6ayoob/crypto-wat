import asyncio
import time
import hmac
import hashlib
import urllib.parse
import logging
import aiohttp
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher
from config import TELEGRAM_TOKEN, ALLOWED_USER_IDS, BYBIT_API_KEY, BYBIT_API_SECRET

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

BYBIT_SPOT_API = "https://api.bybit.com"

RISK_PERCENT = 40  # نسبة الشراء من الرصيد %
STOP_LOSS_PERCENT = 1.5  # وقف خسارة أولي %
TRAILING_STOPS = [1, 3]  # رفع وقف الخسارة عند تحقيق 1% و 3% ربح

# لتخزين حالة الصفقات المفتوحة: { "SYMBOL": {side, qty, entry_price, stop_loss} }
open_positions = {}

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

async def get_symbols():
    # جلب قائمة العملات Spot من Bybit (حسب الوثائق)
    res = await bybit_request("GET", "/spot/v1/symbols")
    if res.get("ret_code") == 0:
        all_symbols = [s['name'] for s in res['result']['list'] if s['quote_currency'] == "USDT"]
        return all_symbols[:50]  # أفضل 50 عملة
    return []

async def get_klines(symbol, interval="60", limit=200):
    # Bybit Spot API: /spot/quote/v1/kline?symbol=BTCUSDT&interval=60&limit=200
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    res = await bybit_request("GET", "/spot/quote/v1/kline", params)
    if res.get("ret_code") == 0:
        data = res["result"]["list"]
        df = pd.DataFrame(data)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.astype({
            'open': 'float', 'high': 'float', 'low': 'float',
            'close': 'float', 'volume': 'float'
        })
        return df
    return pd.DataFrame()

def calculate_ma(df, period):
    return df['close'].rolling(window=period).mean()

def check_crossovers(df):
    # تحقق من تقاطع MA20 & MA50 أو MA50 & MA200 في آخر شمعة
    ma20 = calculate_ma(df, 20)
    ma50 = calculate_ma(df, 50)
    ma200 = calculate_ma(df, 200)

    # شروط شراء:
    # تقاطع صعودي MA20 فوق MA50 (ويكون قبلها MA20 تحت MA50)
    cross_20_50 = (ma20.iloc[-2] < ma50.iloc[-2]) and (ma20.iloc[-1] > ma50.iloc[-1])

    # أو تقاطع صعودي MA50 فوق MA200
    cross_50_200 = (ma50.iloc[-2] < ma200.iloc[-2]) and (ma50.iloc[-1] > ma200.iloc[-1])

    return cross_20_50 or cross_50_200

async def place_order(symbol, side, quantity):
    params = {
        "symbol": symbol,
        "side": side.upper(),
        "type": "MARKET",
        "qty": str(round(quantity, 6)),
        "time_in_force": "GTC",
    }
    res = await bybit_request("POST", "/spot/v1/order", params, private=True)
    if res.get("ret_code") == 0:
        return res["result"]
    else:
        logging.error(f"فشل في تنفيذ أمر {side} على {symbol}: {res}")
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
    params = {
        "symbol": symbol,
        "side": "SELL" if side == "BUY" else "BUY",
        "type": "STOP_LIMIT",
        "qty": str(round(quantity, 6)),
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
            logging.error(f"فشل إرسال رسالة إلى {user_id}: {e}")

async def trading_cycle():
    global open_positions
    symbols = await get_symbols()
    if not symbols:
        logging.error("فشل في جلب قائمة العملات")
        return

    balance = await get_spot_balance("USDT")
    if balance < 10:  # شرط الحد الأدنى للرصيد
        await send_telegram("⚠️ رصيد USDT منخفض جداً، يرجى الإيداع.")
        return

    for symbol in symbols:
        try:
            # إذا هناك صفقة مفتوحة للعملة نتابع فقط وقف الخسارة والبيع
            if symbol in open_positions:
                pos = open_positions[symbol]
                price = (await get_klines(symbol, limit=1))['close'].iloc[-1]
                entry = pos['entry_price']
                qty = pos['qty']
                side = pos['side']

                # تحديث وقف الخسارة الديناميكي (Trailing Stop)
                current_profit_pct = (price - entry) / entry * 100 if side == "BUY" else (entry - price) / entry * 100

                new_stop = None
                for threshold in TRAILING_STOPS:
                    if current_profit_pct >= threshold and (pos.get('last_stop_set', 0) < threshold):
                        # نرفع وقف الخسارة ليكون أقل بنسبة 1% أقل من السعر الحالي (تحمي الربح)
                        new_stop_price = round(price * (1 - STOP_LOSS_PERCENT / 100), 6)
                        await cancel_all_stop_orders(symbol)
                        stop_order = await place_stop_loss_limit(symbol, side, qty, new_stop_price)
                        if stop_order:
                            pos['stop_loss'] = new_stop_price
                            pos['last_stop_set'] = threshold
                            await send_telegram(f"🔄 تم رفع وقف الخسارة لـ {symbol} عند {new_stop_price} USDT")
                        break

                # تحقق وقف الخسارة (سيتم تنفيذه أوتوماتيكيًا من البورصة)
                # تحقق هدف الربح (مثلا 5% أو حسب استراتيجية أخرى)
                target_profit = 5  # ثابت هنا لكن يمكن تعديله

                if current_profit_pct >= target_profit:
                    # بيع لجني الربح
                    order = await place_order(symbol, "SELL", qty)
                    if order:
                        await send_telegram(f"🎉 تم بيع {qty} {symbol} بسعر {price} USDT - هدف الربح تحقق!")
                        del open_positions[symbol]
                        await cancel_all_stop_orders(symbol)
                elif price <= pos['stop_loss']:
                    # السعر وصل وقف الخسارة - البورصة ستنفذ وقف الخسارة أو نبيع مباشرة هنا (لضمان)
                    order = await place_order(symbol, "SELL", qty)
                    if order:
                        await send_telegram(f"⚠️ تم بيع {qty} {symbol} بسعر {price} USDT - وقف الخسارة تحقق!")
                        del open_positions[symbol]
                        await cancel_all_stop_orders(symbol)

            else:
                # لا صفقة مفتوحة، نتحقق شروط الدخول
                df = await get_klines(symbol)
                if df.empty or len(df) < 200:
                    continue

                if check_crossovers(df):
                    price = df['close'].iloc[-1]
                    qty = (balance * RISK_PERCENT / 100) / price
                    if qty * price < 10:
                        # لا نشتري أقل من 10 USDT قيمة
                        continue

                    order = await place_order(symbol, "BUY", qty)
                    if order:
                        stop_loss_price = round(price * (1 - STOP_LOSS_PERCENT / 100), 6)
                        await cancel_all_stop_orders(symbol)
                        stop_order = await place_stop_loss_limit(symbol, "BUY", qty, stop_loss_price)

                        open_positions[symbol] = {
                            "side": "BUY",
                            "qty": qty,
                            "entry_price": price,
                            "stop_loss": stop_loss_price,
                            "last_stop_set": 0,
                        }
                        await send_telegram(f"✅ تم شراء {qty} {symbol} بسعر {price} USDT مع وقف خسارة عند {stop_loss_price}")
        except Exception as e:
            logging.error(f"خطأ في التعامل مع {symbol}: {e}")

async def trading_loop():
    while True:
        await trading_cycle()
        await asyncio.sleep(25 * 60)  # 25 دقيقة

async def main():
    asyncio.create_task(trading_loop())
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())
