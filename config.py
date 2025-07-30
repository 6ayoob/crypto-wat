import os

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))

TRADE_AMOUNT_USDT = float(os.getenv("TRADE_AMOUNT_USDT", 20))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 5))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 3))
