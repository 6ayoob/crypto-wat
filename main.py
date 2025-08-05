import time
import json
from telegram_bot import send_telegram_message

send_telegram_message("🚀 البوت بدأ العمل بنجاح!")

from strategy import enter_trade, check_positions, is_market_bearish, generate_daily_report

TRADE_SYMBOLS = [
    "ATOM-USDT", "CFX-USDT", "ILV-USDT", "ADA-USDT", "XRP-USDT",
    "DOT-USDT", "MATIC-USDT", "LTC-USDT", "LINK-USDT", "PEPE-USDT",
    "XLM-USDT", "VET-USDT", "FIL-USDT", "ICP-USDT", "ALGO-USDT",
    "MANA-USDT", "SAND-USDT", "EOS-USDT", "CHZ-USDT", "XTZ-USDT",
    "NEAR-USDT", "AAVE-USDT", "KSM-USDT", "RUNE-USDT", "ENJ-USDT",
    "ZIL-USDT", "BAT-USDT", "CRV-USDT", "GRT-USDT", "STX-USDT"
]

MAX_POSITIONS = 5
CHECK_INTERVAL = 60 * 60
DAILY_REPORT_HOUR = 15  # 3 عصراً

def run_bot():
    last_report_day = None
    while True:
        now = time.localtime()
        print(f"🕒 بدء الفحص عند: {time.strftime('%Y-%m-%d %H:%M:%S', now)}")
        print("🚀 بدء التحقق من الصفقات وتنفيذ الاستراتيجية")

        # إرسال التقرير اليومي مرة واحدة كل يوم
        if now.tm_hour == DAILY_REPORT_HOUR and (last_report_day != now.tm_yday):
            generate_daily_report()
            last_report_day = now.tm_yday

        # التحقق إذا السوق هابط
        if is_market_bearish(TRADE_SYMBOLS):
            print("⚠️ السوق في حالة هبوط، إيقاف التداول اليوم.")
            time.sleep(CHECK_INTERVAL)
            continue

        # مراقبة الصفقات المفتوحة
        check_positions()

        # تحميل الصفقات المفتوحة
        try:
            with open("positions.json", "r") as f:
                positions = json.load(f)
        except Exception:
            positions = {}

        open_count = len(positions)
        print(f"⚙️ صفقات مفتوحة حالياً: {open_count}, الحد الأقصى: {MAX_POSITIONS}")

        # فتح صفقات جديدة حتى نصل الحد الأقصى
        if open_count < MAX_POSITIONS:
            for symbol in TRADE_SYMBOLS:
                if symbol not in positions:
                    if enter_trade(symbol):
                        open_count += 1
                    if open_count >= MAX_POSITIONS:
                        break

        print(f"⏳ الانتظار لمدة {CHECK_INTERVAL} ثانية قبل التحقق التالي...\n")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_bot()
