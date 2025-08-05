import time
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from strategy import check_signal, execute_buy, manage_position, load_position

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

if __name__ == "__main__":
    send_telegram_message("🤖 بدأ البوت في مراقبة الأسواق!")

    while True:
        try:
            for symbol in SYMBOLS:
                position = load_position(symbol)
                if position is None:
                    signal = check_signal(symbol)
                    if signal == "buy":
                        order, message = execute_buy(symbol)
                        send_telegram_message(message)
                else:
                    manage_position(symbol, send_telegram_message)

        except Exception as e:
            send_telegram_message(f"⚠️ خطأ:\n{str(e)}")

        time.sleep(60)
