import asyncio
import logging
from datetime import datetime, timezone
from pybit import HTTP
import aiohttp
from telegram import Bot
import config

logging.basicConfig(level=logging.INFO)

bot = Bot(token=config.TELEGRAM_TOKEN)

session = HTTP(
    endpoint="https://api.bybit.com",
    api_key=config.API_KEY,
    api_secret=config.API_SECRET,
)

COINGECKO_API = "https://api.coingecko.com/api/v3"

async def fetch_top_100():
    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "price_change_percentage": "5m,15m,1h"
    }
    async with aiohttp.ClientSession() as session_http:
        async with session_http.get(url, params=params) as resp:
            data = await resp.json()
            if isinstance(data, dict) and data.get("error"):
                logging.error(f"CoinGecko API error: {data.get('error')}")
                return []
            return data

last_alerted = {}

def usdt_to_qty(usdt_amount, price):
    return round(usdt_amount / price, 4)

async def send_telegram_message(text):
    await bot.send_message(chat_id=config.TELEGRAM_CHAT_ID, text=text)

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
        logging.info(f"Order placed: {response}")
        return response
    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")
        return None

async def check_signals():
    logging.info("Checking market conditions...")
    coins = await fetch_top_100()
    if not isinstance(coins, list):
        logging.error("Unexpected data format from CoinGecko API")
        return

    now = datetime.now(timezone.utc)

    for coin in coins:
        try:
            symbol = coin['symbol'].upper() + "USDT"
            price = coin['current_price']
            price_change_5m = coin.get('price_change_percentage_5m_in_currency', 0)
            volume_15m = coin.get('total_volume', 0)  # Approximation

            key = coin['id']
            last_time = last_alerted.get(key)
            if price_change_5m and price_change_5m > 2 and (not last_time or (now - last_time).seconds > 1800):
                qty = usdt_to_qty(config.TRADE_AMOUNT_USDT, price)
                order_resp = await place_order(symbol, "Buy", qty)
                if order_resp and 'ret_code' in order_resp and order_resp['ret_code'] == 0:
                    last_alerted[key] = now
                    msg = (
                        f"🚨 تم فتح صفقة شراء على {coin['name']} ({symbol})\\n"
                        f"💰 السعر: ${price}\\n"
                        f"📈 التغير خلال 5 دقائق: {price_change_5m:.2f}%\\n"
                        f"🔢 الكمية: {qty}\\n"
                        f"🎯 هدف الربح: +{config.TAKE_PROFIT_PERCENT}%\\n"
                        f"🛑 وقف الخسارة: -{config.STOP_LOSS_PERCENT}%\\n"
                    )
                    await send_telegram_message(msg)

                    # TODO: إضافة أوامر وقف خسارة وهدف ربح

        except Exception as e:
            logging.error(f"Error processing coin {coin.get('id')}: {e}")

async def main_loop():
    while True:
        try:
            await check_signals()
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(main_loop())
