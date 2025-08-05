from aiogram import Bot, Dispatcher, types
import asyncio
from strategy import enter_trade, check_positions, load_positions

API_TOKEN = "8300868885:AAEx8Zxdkz9CRUHmjJ0vvn6L3kC2kOPCHuk"
ALLOWED_USER_IDS = [658712542]

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

symbols_to_check = [
    "XRP/USDT", "DOGE/USDT", "MATIC/USDT", "LTC/USDT", "ADA/USDT",
    "TRX/USDT", "ETC/USDT", "FIL/USDT", "EOS/USDT", "NEAR/USDT",
    "VET/USDT", "THETA/USDT", "ZRX/USDT", "CHZ/USDT", "CRO/USDT",
    "BAT/USDT", "SAND/USDT", "MANA/USDT", "KAVA/USDT", "ALGO/USDT"
]

@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    if message.from_user.id not in ALLOWED_USER_IDS:
        await message.answer("❌ غير مسموح لك باستخدام هذا البوت.")
        return
    await message.answer("مرحباً! استخدم /scan لفحص العملات، و /positions لعرض الصفقات المفتوحة.")

@dp.message_handler(commands=["scan"])
async def cmd_scan(message: types.Message):
    if message.from_user.id not in ALLOWED_USER_IDS:
        await message.answer("❌ غير مسموح لك باستخدام هذا البوت.")
        return

    await message.answer("⏳ جار فحص العملات وتنفيذ الصفقات...")
    count = 0
    for symbol in symbols_to_check:
        entered = enter_trade(symbol)
        if entered:
            count += 1
    await message.answer(f"✅ انتهى الفحص. تم فتح {count} صفقة جديدة.")

@dp.message_handler(commands=["positions"])
async def cmd_positions(message: types.Message):
    if message.from_user.id not in ALLOWED_USER_IDS:
        await message.answer("❌ غير مسموح لك باستخدام هذا البوت.")
        return

    positions = load_positions()
    if not positions:
        await message.answer("⚠️ لا توجد صفقات مفتوحة حالياً.")
        return

    msg_lines = ["📋 الصفقات المفتوحة:"]
    for sym, pos in positions.items():
        msg_lines.append(f"{sym}: الدخول عند {pos['entry_price']}, حجم {pos['size']}, وقف خسارة {pos['stop_loss']}, هدف ربح {pos['take_profit']}")
    await message.answer("\n".join(msg_lines))

async def periodic_check():
    while True:
        check_positions()
        await asyncio.sleep(300)  # تحقق كل 5 دقائق

async def main():
    from aiogram import executor
    import logging
    logging.basicConfig(level=logging.INFO)

    # Start periodic check in background
    asyncio.create_task(periodic_check())
    # Start bot
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())
