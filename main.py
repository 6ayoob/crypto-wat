import time
import json
from strategy import enter_trade, check_positions, is_market_bearish, generate_daily_report

TRADE_SYMBOLS = [
    "ATOM-USDT", "CFX-USDT", "ILV-USDT", "ADA-USDT", "XRP-USDT",
    "DOT-USDT", "MATIC-USDT", "LTC-USDT", "LINK-USDT", "BCH-USDT",
    "XLM-USDT", "VET-USDT", "FIL-USDT", "ICP-USDT", "ALGO-USDT",
    "MANA-USDT", "SAND-USDT", "EOS-USDT", "CHZ-USDT", "XTZ-USDT",
    "NEAR-USDT", "AAVE-USDT", "KSM-USDT", "RUNE-USDT", "ENJ-USDT",
    "ZIL-USDT", "BAT-USDT", "CRV-USDT", "GRT-USDT", "STX-USDT"
]

MAX_POSITIONS = 5  # عدد الصفقات المفتوحة القصوى
CHECK_INTERVAL = 60 * 60  # التحقق كل ساعة
DAILY_REPORT_HOUR = 12  # الساعة 12 ظهراً بتوقيت السيرفر (يمكن تعديله)

def run_bot():
    last_report_day = None

    while True:
        now = time.localtime()
        print("🚀 بدء التحقق من الصفقات وتنفيذ الاستراتيجية")

        # إرسال التقرير اليومي عند الوقت المحدد
        if now.tm_hour == DAILY_REPORT_HOUR and (last_report_day != now.tm_yday):
            generate_daily_report()
            last_report_day = now.tm_yday

        # التحقق من حالة السوق العام
        if is_market_bearish(TRADE_SYMBOLS):
            print("⚠️ السوق في حالة هبوط، إيقاف التداول اليوم.")
            time.sleep(CHECK_INTERVAL)
            continue  # لا تفتح صفقات جديدة، فقط تحقق من الصفقات المفتوحة

        # تحقق من الصفقات المفتوحة وبيع عند وقف الخسارة/جني الربح
        check_positions()

        # جلب الصفقات المفتوحة من الملف
        try:
            with open("positions.json", "r") as f:
                positions = json.load(f)
        except Exception:
            positions = {}

        open_count = len(positions)

        # افتح صفقات جديدة إذا لم نصل للحد الأقصى
        if open_count < MAX_POSITIONS:
            for symbol in TRADE_SYMBOLS:
                if symbol not in positions:
                    enter_trade(symbol)
                    break  # صفقة واحدة فقط في كل دورة

        print(f"⏳ الانتظار لمدة {CHECK_INTERVAL} ثانية قبل التحقق التالي...\n")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_bot()
