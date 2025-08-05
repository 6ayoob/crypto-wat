import asyncio
from aiogram import Bot, Dispatcher, executor, types
from config import TELEGRAM_TOKEN, ALLOWED_USER_IDS
import strategy

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher(bot)

# السماح فقط لمستخدمين محددين باستخدام البوت
ALLOWED_USER_IDS = [658712542]  # عدل حسب معرفك

async def restricted_access(message: types.Message):
    if message.from_user.id not in ALLOWED_USER_IDS:
        await message.reply("❌ ليس لديك صلاحية استخدام هذا البوت.")
        return False
    return True

@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    if not await restricted_access(message):
        return
    await message.reply("أهلاً! استخدم /scan لفحص السوق، /status لعرض الصفقات المفتوحة، /set_tp و /set_sl لتعديل نسب جني الأرباح ووقف الخسارة.")

@dp.message_handler(commands=['scan'])
async def scan_handler(message: types.Message):
    if not await restricted_access(message):
        return
    await message.reply("🔍 جاري فحص السوق وتنفيذ الصفقات المحتملة...")
    # مثال: فحص رمز واحد أو أكثر (يمكنك تعديل القائمة)
    symbols_to_check = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    count = 0
    for symbol in symbols_to_check:
        if strategy.enter_trade(symbol):
            count += 1
    await message.reply(f"✅ تمت معالجة {count} صفقة.")

@dp.message_handler(commands=['status'])
async def status_handler(message: types.Message):
    if not await restricted_access(message):
        return
    summary = strategy.get_positions_summary()
    await message.reply(summary)

@dp.message_handler(commands=['set_tp'])
async def set_tp_handler(message: types.Message):
    if not await restricted_access(message):
        return
    try:
        percent = float(message.get_args())
        if 1 <= percent <= 10:
            strategy.TAKE_PROFIT_PERCENT = percent
            await message.reply(f"✅ تم تعديل نسبة جني الأرباح إلى {percent}%")
        else:
            await message.reply("❌ أدخل نسبة بين 1 و 10")
    except Exception:
        await message.reply("❌ استخدم الأمر بشكل صحيح: /set_tp 4")

@dp.message_handler(commands=['set_sl'])
async def set_sl_handler(message: types.Message):
    if not await restricted_access(message):
        return
    try:
        percent = float(message.get_args())
        if 0.1 <= percent <= 5:
            strategy.STOP_LOSS_PERCENT = percent
            await message.reply(f"✅ تم تعديل نسبة وقف الخسارة إلى {percent}%")
        else:
            await message.reply("❌ أدخل نسبة بين 0.1 و 5")
    except Exception:
        await message.reply("❌ استخدم الأمر بشكل صحيح: /set_sl 1")

async def periodic_check():
    while True:
        print("🔄 فحص المراكز المفتوحة...")
        strategy.check_positions()
        await asyncio.sleep(300)  # كل 5 دقائق

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(periodic_check())
    executor.start_polling(dp)
