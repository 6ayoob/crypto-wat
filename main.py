import time
import requests
from datetime import datetime, timedelta
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from strategy import check_signal, execute_buy, manage_position, load_position, count_open_positions

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

def generate_daily_report():
    report_lines = []
    open_positions_count = count_open_positions()
    report_lines.append(f"📊 تقرير الصفقات المفتوحة اليوم ({open_positions_count} صفقة):\n")

    if open_positions_count == 0:
        report_lines.append("لا توجد صفقات مفتوحة حالياً.")
        return "\n".join(report_lines)

    for symbol in SYMBOLS:
        pos = load_position(symbol)
        if pos:
            # بيانات المركز
            symbol = pos['symbol']
            entry = pos['entry_price']
            stop = pos['stop_loss']
            take = pos['take_profit']
            amount = pos['amount']
            current_price = None
            try:
                from okx_api import fetch_price
                current_price = fetch_price(symbol)
            except:
                current_price = "N/A"
            line = (
                f"{symbol}:\n"
                f"  السعر الحالي: {current_price}\n"
                f"  سعر الدخول: {entry:.4f}\n"
                f"  وقف الخسارة: {stop:.4f}\n"
                f"  هدف الربح: {take:.4f}\n"
                f"  الكمية: {amount:.6f}\n"
            )
            report_lines.append(line)
    return "\n".join(report_lines)

if __name__ == "__main__":
    send_telegram_message("🤖 بدأ البوت في مراقبة الأسواق!")

    last_report_date = None

    while True:
        try:
            now_utc = datetime.utcnow()
            now_saudi = now_utc + timedelta(hours=3)  # توقيت السعودية UTC+3

            # إرسال التقرير اليومي مرة واحدة عند الساعة 3:00 فجراً
            if now_saudi.hour == 3 and (last_report_date is None or last_report_date != now_saudi.date()):
                report = generate_daily_report()
                send_telegram_message(report)
                last_report_date = now_saudi.date()

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
