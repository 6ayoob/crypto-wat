# main.py

import time
from strategy import check_signal, execute_buy, manage_position, load_position
from okx_api import get_all_symbols
from telegram_bot import send_message

WATCHED_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]

def run_bot():
    send_message("🤖 بدأ تشغيل بوت التداول اللحظي!")

    while True:
        try:
            for symbol in WATCHED_SYMBOLS:
                # إدارة الصفقة إذا كانت موجودة
                if load_position(symbol):
                    manage_position(symbol, send_message)
                    continue

                # إذا لا يوجد صفقة، افحص للإشارة
                signal = check_signal(symbol)

                if signal == "buy":
                    order, msg = execute_buy(symbol)
                    if order:
                        send_message(msg)

            time.sleep(60)  # انتظر دقيقة واحدة قبل الجولة التالية

        except Exception as e:
            send_message(f"❌ حدث خطأ: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_bot()
