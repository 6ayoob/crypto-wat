import time
import json  # <- اضف هذا السطر
from strategy import enter_trade, check_positions

TRADE_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "XRP-USDT",
    "DOT-USDT", "MATIC-USDT", "LTC-USDT", "LINK-USDT", "BCH-USDT",
    "XLM-USDT", "VET-USDT", "FIL-USDT", "ICP-USDT", "ALGO-USDT",
    "MANA-USDT", "SAND-USDT", "EOS-USDT", "CHZ-USDT", "XTZ-USDT",
    "NEAR-USDT", "AAVE-USDT", "KSM-USDT", "RUNE-USDT", "ENJ-USDT",
    "ZIL-USDT", "BAT-USDT", "CRV-USDT", "GRT-USDT", "STX-USDT"
]

MAX_POSITIONS = 5  # عدد الصفقات المفتوحة القصوى
CHECK_INTERVAL = 60 * 60  # ساعة

def run_bot():
    while True:
        print("🚀 تشغيل التحقق من الصفقات وتنفيذ الصفقات الجديدة")

        # تحقق من الصفقات المفتوحة
        check_positions()

        # جلب الصفقات المفتوحة من الملف
        with open("positions.json", "r") as f:
            positions = json.load(f)

        # كم صفقة مفتوحة؟
        open_count = len(positions)

        # فتح صفقات جديدة إذا لم نصل للحد الأقصى
        if open_count < MAX_POSITIONS:
            for symbol in TRADE_SYMBOLS:
                if symbol not in positions:
                    enter_trade(symbol)
                    break  # افتح صفقة واحدة فقط في كل دورة

        print(f"⏳ انتظار {CHECK_INTERVAL} ثانية قبل التحقق التالي...\n")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    run_bot()
