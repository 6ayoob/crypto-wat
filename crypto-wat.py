import asyncio
import logging
from datetime import datetime, timezone
from pybit.usdt_perpetual import HTTP
import aiohttp
from telegram import Bot
import config

logging.basicConfig(level=logging.INFO)

bot = Bot(token=config.TELEGRAM_TOKEN)

session = HTTP(
    endpoint="https://api.bybit.com",
    api_key=config.API_KEY,
    api_secret=config.API_SECRET,
    recv_window=10000
)

COINGECKO_API = "https://api.coingecko.com/api/v3"
last_alerted = {}

def usdt_to_qty(usdt_amount, price):
    return round(usdt_amount / price, 4)

async def send_telegram_message(text):
    try:
        await bot.send_message(chat_id=config.TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        logging.error(f"❌ فشل في إرسال رسالة تيليجرام: {e}")

async def fetch_top_100():
    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "price_change_percentage": "5m,15m,1h"
    }

    for attempt in range(3):
        async with aiohttp.ClientSession() as session_http:
            try:
                async with session_http.get(url, params=params) as resp:
                    if resp.status == 429:
                        logging.warning("🚫 تجاوزت الحد المسموح من CoinGecko! سيتم الانتظار 60 ثانية...")
                        await asyncio.sleep(60)
                        continue
                    elif resp.status != 200:
                        logging.error(f"❌ خطأ من CoinGecko: {resp.status}")
                        return []
                    data = await resp.json()
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logging.exception(f"❌ استثناء أثناء جلب البيانات من CoinGecko: {e}")
                await asyncio.sleep(5)
    return []

async def place_order(symbol, side, qty):
    try:
        response = session.place_active_order(
            symbol=symbol,
            side=side,
            order_type="Market",
            qty=qty,
            time_in_force="GoodTillCancel",
            reduce_only=False,
            close_on_trigger=False,
        )
        logging.info(f"✅ تم تنفيذ الطلب: {response}")
        return response
    except Exception as e:
        logging.error(f"❌ خطأ في تنفيذ الطلب على {symbol}: {e}")
        return None

async def check_signals():
    logging.info("🔍 جاري فحص السوق...")
    coins = await fetch_top_100()
    if not coins:
        return

    now = datetime.now(timezone.utc)

    for coin in coins:
        try:
            symbol = coin['symbol'].upper() + "USDT"
            price = coin['current_price']
            price_change_5m = coin.get('price_change_percentage_5m_in_currency', 0)

            key = coin['id']
            last_time = last_alerted.get(key)
            if price_change_5m and price_change_5m > 2 and (not last_time or (now - last_time).seconds > 1800):
                qty = usdt_to_qty(config.TRADE_AMOUNT_USDT, price)
                order_resp = await place_order(symbol, "Buy", qty)
                if order_resp and order_resp.get('ret_code') == 0:
                    last_alerted[key] = now
                    msg = (
                        f"🚨 تم فتح صفقة شراء على {coin['name']} ({symbol})\n"
                        f"💰 السعر: ${price}\n"
                        f"📈 التغير خلال 5 دقائق: {price_change_5m:.2f}%\n"
                        f"🔢 الكمية: {qty}\n"
                        f"🎯 هدف الربح: +{config.TAKE_PROFIT_PERCENT}%\n"
                        f"🛑 وقف الخسارة: -{config.STOP_LOSS_PERCENT}%\n"
                    )
                    await send_telegram_message(msg)

        except Exception as e:
            logging.error(f"❌ خطأ أثناء معالجة العملة {coin.get('id')}: {e}")

async def test_connections():
    try:
        await bot.send_message(chat_id=config.TELEGRAM_CHAT_ID, text="✅ البوت يعمل وتستطيع استقبال الرسائل")
    except Exception as e:
        logging.error(f"❌ خطأ في اختبار الاتصال: {e}")

async def main_loop():
    await test_connections()
    while True:
        try:
            await check_signals()
        except Exception as e:
            logging.error(f"❌ خطأ في حلقة التنفيذ الرئيسية: {e}")
        await asyncio.sleep(900)  # كل 15 دقيقة

# استخدام طريقة مستقرة لتشغيل اللوب
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_loop())
