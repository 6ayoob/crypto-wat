import time
from datetime import datetime
from strategy import analyze_symbol
from okx_api import get_balance, get_last_price, place_market_order
import requests

TELEGRAM_TOKEN = "توكن_بوتك"
TELEGRAM_CHAT_ID = "معرفك"

WATCHLIST = ["ATOM-USDT", "SOL-USDT", "INJ-USDT", "MATIC-USDT"]
open_trades = {}
MARKET_DROP = False

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, data=data)
    except:
        pass

def check_market_health():
    btc_data = analyze_symbol("BTC-USDT")
    if not btc_data:
        return False
    return btc_data["rsi"] > 40 and btc_data["price_trend"] == "UP"

def run_strategy():
    global MARKET_DROP
    if not check_market_health():
        if not MARKET_DROP:
            send_telegram("⚠️ تم إيقاف التداول مؤقتًا بسبب هبوط السوق (BTC تحت الضغط)")
            MARKET_DROP = True
        return
    else:
        if MARKET_DROP:
            send_telegram("✅ تم استئناف التداول بعد تعافي السوق")
            MARKET_DROP = False

    usdt_balance = get_balance()
    per_trade_amount = usdt_balance / len(WATCHLIST)

    for symbol in WATCHLIST:
        if symbol in open_trades:
            entry = open_trades[symbol]["entry_price"]
            last = get_last_price(symbol)
            stop_price = open_trades[symbol]["stop_loss"]

            # Trailing Stop Loss
            if last > entry * 1.05:
                open_trades[symbol]["stop_loss"] = max(stop_price, last * 0.97)

            # تحقق من ضرب وقف الخسارة
            if last < open_trades[symbol]["stop_loss"]:
                place_market_order(symbol, "sell", open_trades[symbol]["qty"])
                send_telegram(f"❌ تم بيع {symbol} بـ {last:.3f} (وقف خسارة متحرك)")
                del open_trades[symbol]
            continue

        analysis = analyze_symbol(symbol)
        if analysis:
            price = get_last_price(symbol)
            qty = round(per_trade_amount / price, 4)
            place_market_order(symbol, "buy", qty)
            open_trades[symbol] = {
                "entry_price": price,
                "qty": qty,
                "stop_loss": price * 0.97
            }
            send_telegram(f"✅ تم شراء {symbol} بسعر {price:.3f} وحجم {qty}")

def send_daily_report():
    if not open_trades:
        send_telegram("📊 لا توجد صفقات مفتوحة حاليًا.")
        return
    msg = "📊 تقرير يومي للصفقات:\n"
    for sym, data in open_trades.items():
        current = get_last_price(sym)
        change = (current - data["entry_price"]) / data["entry_price"] * 100
        msg += f"\n{sym}:\nالشراء: {data['entry_price']:.3f} | الآن: {current:.3f} | {change:.2f}%"
    send_telegram(msg)

if __name__ == "__main__":
    while True:
        now = datetime.now()
        run_strategy()

        # إرسال التقرير الساعة 3 عصرًا فقط مرة واحدة
        if now.hour == 15 and now.minute == 0:
            send_daily_report()
            time.sleep(60)

        time.sleep(30)
