import asyncio
import logging
import aiohttp
from aiogram import Bot, Dispatcher, types
from config import TELEGRAM_TOKEN, ALLOWED_USER_IDS
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO)

COINGECKO_API = "https://api.coingecko.com/api/v3"

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

cg_cache = {
    "data": None,
    "timestamp": None
}


async def fetch_top_100(retries=3):
    now = datetime.now(timezone.utc)

    if cg_cache["data"] and cg_cache["timestamp"]:
        if (now - cg_cache["timestamp"]).total_seconds() < 60:
            return cg_cache["data"]

    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "price_change_percentage": "5m,15m,1h"
    }

    for attempt in range(retries):
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
                        cg_cache["data"] = data
                        cg_cache["timestamp"] = now
                        return data
                    else:
                        logging.error(f"❌ تنسيق البيانات غير متوقع: {data}")
                        return []
            except Exception as e:
                logging.exception(f"❌ استثناء أثناء جلب البيانات من CoinGecko: {e}")
                await asyncio.sleep(5)
    return []


async def scan_strategy_1():
    """
    الاستراتيجية 1: ارتفاع مفاجئ في السعر (+2% خلال 15 دقيقة) وحجم تداول عالي (>5M).
    """
    logging.info("🔍 تنفيذ الاستراتيجية 1: ارتفاع السعر المفاجئ والحجم العالي...")
    coins = await fetch_top_100()
    movers = []

    for coin in coins:
        price = coin.get("current_price", 0)
        volume = coin.get("total_volume", 0)
        change_15m = coin.get("price_change_percentage_15m_in_currency", 0)

        if price > 0 and volume > 5_000_000 and change_15m and change_15m > 2:
            movers.append(f"{coin['symbol'].upper()} ⏫ {change_15m:.2f}% (حجم: {volume})")

    if movers:
        return "📈 [استراتيجية 1] العملات النشطة:\n" + "\n".join(movers)
    else:
        return "🔎 [استراتيجية 1] لا توجد حركات بارزة حاليًا."


async def scan_strategy_2():
    """
    الاستراتيجية 2: ارتفاع مفاجئ في حجم التداول (+50% عن المتوسط) مع تغير بسيط في السعر (±1%).
    """
    logging.info("🔍 تنفيذ الاستراتيجية 2: ارتفاع حجم التداول المفاجئ وتغير بسيط في السعر...")
    coins = await fetch_top_100()
    movers = []

    for coin in coins:
        price = coin.get("current_price", 0)
        volume = coin.get("total_volume", 0)
        change_15m = coin.get("price_change_percentage_15m_in_currency", 0)
        # للحصول على المتوسط، نفترض أن لدينا متوسط حجم التداول محفوظ في coin.get("average_volume_50d")
        # لكن CoinGecko لا يعطي هذا، لذلك هذه نقطة لتحسين لاحقًا.
        # الآن نستخدم قيمة ثابتة تقريبية 3,000,000 كمتوسط افتراضي (يمكن تعديلها حسب الحاجة).
        average_volume = 3_000_000

        if price > 0 and volume > average_volume * 1.5 and change_15m is not None and -1 <= change_15m <= 1:
            movers.append(f"{coin['symbol'].upper()} 🔄 حجم تداول مرتفع مع تغير سعر بسيط ({change_15m:.2f}%)")

    if movers:
        return "📊 [استراتيجية 2] العملات ذات حجم تداول مرتفع:\n" + "\n".join(movers)
    else:
        return "🔎 [استراتيجية 2] لا توجد حركات بارزة حاليًا."


@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message):
    if str(message.from_user.id) not in ALLOWED_USER_IDS:
        return
    await message.answer("🤖 أهلاً بك! أرسل /scan لعرض العملات النشطة حالياً.")


@dp.message_handler(commands=["scan"])
async def handle_scan(message: types.Message):
    if str(message.from_user.id) not in ALLOWED_USER_IDS:
        return
    # نفحص باستخدام الاستراتيجية 1 عند الطلب كافتراض
    report = await scan_strategy_1()
    await message.answer(report)


async def main_loop():
    strategy_duration = timedelta(hours=3)
    current_strategy = 1
    strategy_start_time = datetime.now(timezone.utc)

    while True:
        now = datetime.now(timezone.utc)
        elapsed = now - strategy_start_time

        if elapsed > strategy_duration:
            # نبدل الاستراتيجية
            current_strategy = 2 if current_strategy == 1 else 1
            strategy_start_time = now
            logging.info(f"🔄 تبديل إلى الاستراتيجية {current_strategy}")

        if current_strategy == 1:
            report = await scan_strategy_1()
        else:
            report = await scan_strategy_2()

        for user_id in ALLOWED_USER_IDS:
            try:
                await bot.send_message(chat_id=user_id, text=report)
                await asyncio.sleep(1)  # تأخير بسيط بين الرسائل
            except Exception as e:
                logging.warning(f"❗ خطأ في الإرسال للمستخدم {user_id}: {e}")

        await asyncio.sleep(30)  # تحقق كل 30 ثانية هل الوقت لتبديل الاستراتيجية وتنفيذها مرة أخرى


async def main():
    asyncio.create_task(main_loop())
    await dp.start_polling()


if __name__ == "__main__":
    asyncio.run(main())
