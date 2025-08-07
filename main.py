import time
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from strategy import check_signal, execute_buy, manage_position, load_position
import requests

MAX_OPEN_POSITIONS = 3  # الحد الأقصى للصفقات المفتوحة في نفس الوقت

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

if __name__ == "__main__":
    send_telegram_message("🚀 وماتوفيقي بالا بالله رب العالمين  EMA9/EMA21 + RSI مع هدف واحد ووقف خسارة ✅")

    while True:
        try:
            open_positions_count = 0

            # احسب عدد الصفقات المفتوحة حاليًا
            for symbol in SYMBOLS:
                if load_position(symbol) is not None:
                    open_positions_count += 1

            for symbol in SYMBOLS:
                position = load_position(symbol)

                if position is None:
                    if open_positions_count >= MAX_OPEN_POSITIONS:
                        continue  # تجاهل الفتح إذا وصلنا للحد الأقصى
                    signal = check_signal(symbol)
                    if signal == "buy":
                        order, message = execute_buy(symbol)
                        if message:
                            send_telegram_message(message)
                        if order:
                            open_positions_count += 1  # زيادة العداد بعد فتح صفقة
                else:
                    closed = manage_position(symbol)
                    if closed:
                        send_telegram_message(f"صفقة {symbol} أُغلقت بناءً على هدف الربح أو وقف الخسارة.")
                        open_positions_count -= 1  # تقليل العداد إذا أغلقت الصفقة

        except Exception as e:
            import traceback
            send_telegram_message(f"⚠️ خطأ في البوت:\n{traceback.format_exc()}")

        time.sleep(60)
