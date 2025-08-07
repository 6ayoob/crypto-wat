import time
import requests
from datetime import datetime, timedelta
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from strategy import check_signal, execute_buy, manage_position, load_position, count_open_positions, load_closed_positions
from okx_api import fetch_price

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
