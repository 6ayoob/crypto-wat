import pandas as pd
import numpy as np
from okx_api import get_historical_candles

def analyze_trend(instId):
    candles = get_historical_candles(instId, bar="1D", limit=30)
    if not candles:
        return None

    # تحويل البيانات إلى DataFrame وتحديد الأعمدة التسعة حسب OKX
    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "volCcy", "volCcyQuote", "confirm"
    ])

    # تحويل الأعمدة الرقمية إلى float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # حساب متوسط الحركة البسيط لـ 20 يوم
    df["sma20"] = df["close"].rolling(window=20).mean()

    # إشارة دخول: السعر الحالي أكبر من متوسط 20 يوم ودخول في ترند صاعد
    last_close = df["close"].iloc[-1]
    last_sma20 = df["sma20"].iloc[-1]

    if last_close > last_sma20:
        return {
            "trend": "up",
            "last_close": last_close,
            "sma20": last_sma20,
            "recommendation": "buy"
        }
    else:
        return {
            "trend": "down",
            "last_close": last_close,
            "sma20": last_sma20,
            "recommendation": "wait"
        }
