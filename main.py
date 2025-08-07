import time
import requests
from datetime import datetime
import strategy
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

def main_loop():
    while True:
        for symbol in SYMBOLS:
            signal = strategy.check_signal(symbol)
            if signal == "buy":
                order, msg = strategy.execute_buy(symbol)
                send_telegram_message(msg)
            strategy.manage_position(symbol, send_telegram_message)

        time.sleep(60)  # انتظر دقيقة قبل التكرار

if __name__ == "__main__":
    print("تشغيل البوت...")
    main_loop()
