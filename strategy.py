import pandas as pd
import numpy as np
from okx_api import get_historical_candles, get_last_price

def calculate_rsi(data, period=14):
    if len(data) < period:
        return 50  # محايد إذا البيانات قليلة
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def analyze_symbol(symbol):
    raw_data = get_historical_candles(symbol, bar="1D", limit=50)
    if not raw_data or len(raw_data) < 20:
        return None  # لا يوجد بيانات كافية

    # التأكد من الأعمدة
    df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.astype(float)
    df['close'] = df['close']
    df['volume'] = df['volume']

    # RSI
    rsi = calculate_rsi(df)

    # متوسط حجم التداول
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]

    # اتجاه السعر (مقارنة الإغلاق الحالي بمتوسط 20 يوم)
    ma20 = df['close'].rolling(window=20).mean().iloc[-1]
    last_price = df['close'].iloc[-1]
    price_trend = last_price > ma20

    # شرط دخول
    if rsi < 70 and rsi > 30 and current_volume > avg_volume and price_trend:
        return {
            "symbol": symbol,
            "rsi": round(rsi, 2),
            "volume": round(current_volume, 2),
            "price_trend": "UP",
            "score": 1  # تم اجتياز جميع الشروط
        }

    return None
