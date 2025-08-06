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
    report_lines.append(f"📊 تقرير الصفقات المفتوحة ({open_positions_count} صفقة):\n")

    if open_positions_count == 0:
        report_lines.append("✅ لا توجد صفقات مفتوحة حالياً.")
        return "\n".join(report_lines)

    for symbol in SYMBOLS:
        pos = load_position(symbol)
        if pos:
            try:
                from okx_api import fetch_price
                current_price = fetch_price(symbol)
            except:
                current_price = "N/A"

            # قراءة البيانات الجديدة (TP1 / TP2 / Trailing)
            entry = pos.get('entry_price', 0)
            stop = pos.get('stop_loss', 0)
            tp1 = pos.get('tp1', 0)
            tp2 = pos.get('tp2', 0)
            amount = pos.get('amount', 0)
            tp1_hit = pos.get('tp1_hit', False)
            trailing = pos.get('trailing_active', False)

            line = (
                f"{symbol}:\n"
                f"  📈 السعر الحالي: {current_price}\n"
                f"  💰 سعر الدخول: {entry:.4f}\n"
                f"  🛑 وقف الخسارة: {stop:.4f}\n"
                f"  🎯 TP1: {tp1:.4f} {'(✅ محقق)' if tp1_hit else ''}\n"
                f"  🏆 TP2: {tp2:.4f}\n"
                f"  📦 الكمية المتبقية: {amount:.6f}\n"
                f"  📌 Trailing Stop: {'✅ مفعل' if trailing else '❌ غير مفعل'}\n"
            )
            report_lines.append(line)

    return "\n".join(report_lines)

if __name__ == "__main__":
    send_telegram_message("🚀 بدأ البوت بمراقبة الأسواق باستخدام استراتيجية TP1/TP2 + Trailing ✅")

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

            # ✅ مراقبة وإدارة جميع الرموز
            for symbol in SYMBOLS:
                position = load_position(symbol)

                if position is None:
                    signal = check_signal(symbol)
                    if signal == "buy":
                        order, message = execute_buy(symbol)
                        if message:
                            send_telegram_message(message)
                else:
                    manage_position(symbol, send_telegram_message)

        except Exception as e:
            send_telegram_message(f"⚠️ خطأ في main.py:\n{str(e)}")

        time.sleep(60)  # التشغيل كل دقيقة
