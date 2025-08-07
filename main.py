import time
from datetime import datetime, timedelta
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from strategy import check_signal, execute_buy, manage_position, load_position, count_open_positions, load_closed_positions
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

def generate_daily_report():
    report_lines = []
    open_positions_count = count_open_positions()
    report_lines.append(f"📊 تقرير الصفقات المفتوحة ({open_positions_count} صفقة):\n")

    if open_positions_count == 0:
        report_lines.append("✅ لا توجد صفقات مفتوحة حالياً.")
    else:
        for symbol in SYMBOLS:
            pos = load_position(symbol)
            if pos:
                try:
                    current_price = fetch_price(symbol)
                except:
                    current_price = "N/A"

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

    closed_positions = load_closed_positions()
    now = datetime.utcnow()
    recent_closed = [pos for pos in closed_positions if datetime.fromisoformat(pos['closed_at']) > now - timedelta(days=1)]

    if recent_closed:
        report_lines.append("\n📈 صفقات مغلقة خلال آخر 24 ساعة:")
        total_profit = 0
        for pos in recent_closed:
            profit_pct = (pos['profit'] / (pos['entry_price'] * pos['amount'])) * 100 if pos['entry_price'] > 0 else 0
            total_profit += pos['profit']
            report_lines.append(
                f"{pos['symbol']} | دخول: {pos['entry_price']:.4f} خروج: {pos['exit_price']:.4f} | ربح: {pos['profit']:.4f} USDT ({profit_pct:.2f}%)"
            )
        report_lines.append(f"\n🔔 إجمالي الربح/الخسارة خلال 24 ساعة: {total_profit:.4f} USDT")

    return "\n".join(report_lines)

if __name__ == "__main__":
    send_telegram_message("🚀 بدأ البوت بمراقبة الأسواق باستخدام استراتيجية مبسطة وآمنة ✅")
    last_report_date = None

    while True:
        try:
            now_utc = datetime.utcnow()
            now_saudi = now_utc + timedelta(hours=3)

            # إرسال التقرير اليومي الساعة 3 صباحاً بتوقيت السعودية مرة واحدة يومياً
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
                        if message:
                            send_telegram_message(message)
                else:
                    manage_position(symbol, send_telegram_message)

        except Exception as e:
            import traceback
            send_telegram_message(f"⚠️ خطأ في main.py:\n{traceback.format_exc()}")

        time.sleep(60)
