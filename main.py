# main.py

import time
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from strategy import check_signal, execute_buy, manage_position

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})

if __name__ == "__main__":
    send_telegram_message("🤖 بدأ البوت في العمل!")

    while True:
        try:
            signal = check_signal()
            if signal == "buy":
                order, message = execute_buy()
                send_telegram_message(message)

            manage_position(send_telegram_message)

        except Exception as e:
            send_telegram_message(f"⚠️ خطأ في النظام:\n{str(e)}")

        time.sleep(30)
