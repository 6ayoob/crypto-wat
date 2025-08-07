import time
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from strategy import check_signal, execute_buy, manage_position, load_position, count_open_positions
from okx_api import fetch_price
import requests

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

def main_loop():
    print("🚀 بدء تشغيل البوت ...")
    while True:
        for symbol in SYMBOLS:
            signal = check_signal(symbol)
            if signal == "buy":
                order, msg = execute_buy(symbol)
                send_telegram_message(msg)
            # إدارة الصفقة المفتوحة للرمز
            manage_position(symbol, send_telegram_message)
        # انتظر 60 ثانية بين كل دورة
        time.sleep(60)

if __name__ == "__main__":
    main_loop()
