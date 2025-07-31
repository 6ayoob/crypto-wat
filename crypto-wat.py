import asyncio
import logging
import aiohttp
from aiogram import Bot, Dispatcher, types
from config import TELEGRAM_TOKEN, ALLOWED_USER_IDS
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)

COINGECKO_API = "https://api.coingecko.com/api/v3"

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

# تخزين البيانات مؤقتًا لتقليل الضغط
cg_cache = {
    "data": None,
    "timestamp": None
}


async def fetch_top_100():
    now = datetime.now(timezone.utc)

    if cg_cache["data"] and cg_cache["timestamp"]:
        if (now - cg_cache["timestamp"]).total_seconds() < 600:
            return cg_cache["data"]

    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "price_change_percentage": "5m,15m,1h"
    }

    async with aiohttp.ClientSession() as session_http:
        try:
            async with session_http.get(url, params=params) as resp:
                if resp.status == 429:
                    logging.warning("🚫 تجاوزت الحد المسموح من CoinGecko! سيتم الانتظار 60 ثانية...")
                    await asyncio.sleep(60)
                    return []

                elif resp.status != 200:
                    logging.error(f"❌ خطأ من CoinGecko: {resp.status}")
                    return []

                data = await resp.json()
                if isinstance(data, list):
                    cg_cache["data"] = data
                    cg_cache["timestamp"] = now
                    return data
                else:
                    logging.error(f"❌ تنسيق البيانات غير متوقع: {data}")
                    return []

        except Exception as e:
            logging.exception(f"❌ استثناء أثناء جلب البيانات من CoinGecko: {e}")
            return []


async def scan_market():
    logging.info("🔍 جاري فحص السوق...")
    coins = await fetch_top_100()
    movers = []

    for coin in coins:
        price = coin.get("current_price", 0)
        volume = coin.get("total_volume", 0)
        change_15m = coin.get("price_change_percentage_15m_in_currency", 0)

        if price > 0 and volume > 5000000 and change_15m and change_15m > 2:
            movers.append(f"{coin['symbol'].upper()} ⏫ {change_15m:.2f}%")

    if movers:
        return "📈 العملات النشطة:\n" + "\n".join(movers)
    else:
        return "🔎 لا توجد حركات بارزة حاليًا."


@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message):
    if str(message.from_user.id) not in ALLOWED_USER_IDS:
        return
    await message.answer("🤖 أهلاً بك! أرسل /scan لعرض العملات النشطة حالياً.")


@dp.message_handler(commands=["scan"])
async def handle_scan(message: types.Message):
    if str(message.from_user.id) not in ALLOWED_USER_IDS:
        return
    report = await scan_market()
    await message.answer(report)


async def main_loop():
    while True:
        try:
            report = await scan_market()
            for user_id in ALLOWED_USER_IDS:
                try:
                    await bot.send_message(chat_id=user_id, text=report)
                except Exception as e:
                    logging.warning(f"❗ خطأ في الإرسال للمستخدم {user_id}: {e}")
        except Exception as e:
            logging.error(f"❌ خطأ في الحلقة الرئيسية: {e}")

        await asyncio.sleep(1800)  # كل 30 دقيقة


async def main():
    asyncio.create_task(main_loop())
    await dp.start_polling()


if __name__ == "__main__":
    asyncio.run(main())
