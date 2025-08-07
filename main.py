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

def main_loop():
    while True:
        for symbol in SYMBOLS:
            # تحقق هل هناك صفقة مفتوحة حالياً
            position = load_position(symbol)

            if position:
                # إدارة الصفقة المفتوحة (متابعة TP و SL)
                manage_position(symbol, send_telegram_message)
            else:
                # تحقق الإشارة للشراء
                signal = check_signal(symbol)
                if signal == "buy":
                    order, msg = execute_buy(symbol)
                    send_telegram_message(msg)

        # انتظر 60 ثانية بين كل دورة
        time.sleep(60)

if __name__ == "__main__":
    print("🚀 بدء بوت التداول التلقائي...")
    main_loop()
