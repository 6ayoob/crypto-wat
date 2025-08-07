import time
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from strategy import check_signal, execute_buy, manage_position, load_position
import requests

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

if __name__ == "__main__":
    send_telegram_message("🚀 بدأ البوت بمراقبة الأسواق باستخدام استراتيجية EMA9/EMA21 + RSI مع هدف واحد ووقف خسارة ✅")

    while True:
        try:
            for symbol in SYMBOLS:
                position = load_position(symbol)

                if position is None:
                    signal = check_signal(symbol)
                    if signal == "buy":
                        order, message = execute_buy(symbol)
                        if message:
                            send_telegram_message(message)
                else:
                    closed = manage_position(symbol)
                    if closed:
                        send_telegram_message(f"صفقة {symbol} أُغلقت بناءً على هدف الربح أو وقف الخسارة.")

        except Exception as e:
            import traceback
            send_telegram_message(f"⚠️ خطأ في البوت:\n{traceback.format_exc()}")

        time.sleep(60)
