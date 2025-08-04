import time
import json
from strategy import enter_trade, check_positions

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

def run_bot():
    while True:
        print("🚀 بدء التحقق من الصفقات وتنفيذ الاستراتيجية")

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
