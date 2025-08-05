# main.py

import time
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from strategy import check_signal, execute_buy, manage_position

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})

if __name__ == "__main__":
    send_telegram_message("🤖 بدأ البوت في مراقبة الأسواق!")

    while True:
        try:
            for symbol in SYMBOLS:
                signal = check_signal(symbol)
                if signal == "buy":
                    order, message = execute_buy(symbol)
                    send_telegram_message(message)

                manage_position(symbol, send_telegram_message)

        except Exception as e:
            send_telegram_message(f"⚠️ خطأ:\n{str(e)}")

        time.sleep(30)
