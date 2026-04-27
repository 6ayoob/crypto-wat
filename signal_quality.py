# -*- coding: utf-8 -*-
# signal_quality.py - طبقة جودة الإشارة (المرحلة ٢)
#
# الفلاتر الاحترافية الجديدة:
# [Q1] Market Structure: BOS و CHoCH
# [Q2] Order Flow: الحجم يؤكد الاتجاه على 3 شموع
# [Q3] فلتر الوقت: تجنب آخر 20% من الشمعة
# [Q4] فلتر الارتباط: لا صفقتين متشابهتين
# [Q5] فلتر السيولة المتراكمة: أين تجمّع السيولة

from __future__ import annotations

import os, math, time, logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger("signal_quality")

RIYADH_TZ = timezone(timedelta(hours=3))

def _ef(name, default):
    try:    return float(os.getenv(name, str(default)))
    except: return float(default)

def _eb(name, default=False):
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","on")

# ===== إعدادات =====
Q_BOS_ENABLE        = _eb("Q_BOS_ENABLE",        True)
Q_ORDERFLOW_ENABLE  = _eb("Q_ORDERFLOW_ENABLE",  True)
Q_TIME_ENABLE       = _eb("Q_TIME_ENABLE",        True)
Q_CORR_ENABLE       = _eb("Q_CORR_ENABLE",        True)
Q_LIQUIDITY_ENABLE  = _eb("Q_LIQUIDITY_ENABLE",  True)

Q_BOS_LOOKBACK      = int(_ef("Q_BOS_LOOKBACK",   10))   # عدد الشموع للبحث عن هيكل
Q_ORDERFLOW_BARS    = int(_ef("Q_ORDERFLOW_BARS",  3))    # شموع لتأكيد الحجم
Q_TIME_PCT_REMAIN   = _ef("Q_TIME_PCT_REMAIN",   0.25)   # تجنب آخر 25% من الشمعة
Q_CORR_MAX_SYMBOLS  = int(_ef("Q_CORR_MAX_SYMBOLS", 3))  # حد أقصى للرموز المتشابهة

# ══════════════════════════════════════════════════════
# [Q1] Market Structure — BOS و CHoCH
# ══════════════════════════════════════════════════════

def check_market_structure(df) -> Dict[str, Any]:
    """
    يكتشف:
    - BOS (Break of Structure): كسر أعلى قمة سابقة = صعود
    - CHoCH (Change of Character): تغيير الاتجاه
    - HH/HL: Higher Highs / Higher Lows = ترند صاعد

    يرجع:
    {
        "is_bullish_bos": bool,
        "is_choch": bool,
        "structure": "bullish" | "bearish" | "neutral",
        "score_bonus": int,  ← نقاط إضافية للـ score
        "reason": str
    }
    """
    result = {
        "is_bullish_bos": False,
        "is_choch": False,
        "structure": "neutral",
        "score_bonus": 0,
        "reason": ""
    }

    try:
        if len(df) < Q_BOS_LOOKBACK + 5:
            return result

        # آخر N شمعة (بدون الشمعة الحالية)
        seg = df.iloc[-(Q_BOS_LOOKBACK+2):-1]
        highs  = seg["high"].tolist()
        lows   = seg["low"].tolist()
        closes = seg["close"].tolist()

        if len(highs) < 6:
            return result

        # إيجاد القمم والقيعان الهيكلية
        struct_highs = []
        struct_lows  = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                struct_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                struct_lows.append((i, lows[i]))

        if not struct_highs or not struct_lows:
            return result

        last_close    = closes[-1]
        last_high     = highs[-1]
        prev_high_val = struct_highs[-1][1] if struct_highs else 0
        prev_low_val  = struct_lows[-1][1]  if struct_lows  else 0

        # BOS صاعد: كسر أعلى قمة هيكلية سابقة
        if last_close > prev_high_val and last_high > prev_high_val:
            result["is_bullish_bos"] = True
            result["structure"]      = "bullish"
            result["score_bonus"]    = 15
            result["reason"]         = f"BOS↑ كسر {prev_high_val:.4f}"

        # HH/HL: Higher Highs و Higher Lows (ترند صاعد)
        elif len(struct_highs) >= 2 and len(struct_lows) >= 2:
            hh = struct_highs[-1][1] > struct_highs[-2][1]
            hl = struct_lows[-1][1]  > struct_lows[-2][1]
            if hh and hl:
                result["structure"]   = "bullish"
                result["score_bonus"] = 8
                result["reason"]      = "HH+HL هيكل صاعد"

        # CHoCH: السعر كسر أدنى قاع سابق بعد ترند صاعد
        if len(struct_lows) >= 2:
            if last_close < prev_low_val and struct_highs:
                result["is_choch"] = True
                result["reason"]   = f"CHoCH تحذير كسر {prev_low_val:.4f}"
                result["score_bonus"] = -10  # خصم من السكور

    except Exception as e:
        logger.debug(f"[Q1] BOS error: {e}")

    return result


# ══════════════════════════════════════════════════════
# [Q2] Order Flow — تأكيد الحجم على 3 شموع
# ══════════════════════════════════════════════════════

def check_order_flow(df) -> Dict[str, Any]:
    """
    يتحقق من:
    - تراكم الحجم: آخر 3 شموع حجمها يزيد
    - نسبة الشموع الصاعدة (Buy Pressure)
    - هل الحجم الأعلى مع الشموع الصاعدة؟

    يرجع:
    {
        "buy_pressure": float (0-1),
        "volume_trend": "increasing" | "decreasing" | "flat",
        "confirmed": bool,
        "score_bonus": int,
        "reason": str
    }
    """
    result = {
        "buy_pressure":  0.5,
        "volume_trend":  "flat",
        "confirmed":     False,
        "score_bonus":   0,
        "reason":        ""
    }

    try:
        if len(df) < Q_ORDERFLOW_BARS + 5:
            return result

        # آخر N شمعة مغلقة
        seg = df.iloc[-(Q_ORDERFLOW_BARS+2):-1]
        vols   = seg["volume"].tolist()
        opens  = seg["open"].tolist()
        closes = seg["close"].tolist()

        if len(vols) < 3:
            return result

        # حجم الشموع الصاعدة مقابل الهابطة
        bull_vol = sum(vols[i] for i in range(len(vols))
                       if closes[i] > opens[i])
        bear_vol = sum(vols[i] for i in range(len(vols))
                       if closes[i] < opens[i])
        total_vol = bull_vol + bear_vol
        buy_pressure = bull_vol / total_vol if total_vol > 0 else 0.5

        # اتجاه الحجم
        if len(vols) >= 3:
            if vols[-1] > vols[-2] > vols[-3]:
                vol_trend = "increasing"
            elif vols[-1] < vols[-2] < vols[-3]:
                vol_trend = "decreasing"
            else:
                vol_trend = "flat"
        else:
            vol_trend = "flat"

        result["buy_pressure"] = round(buy_pressure, 3)
        result["volume_trend"] = vol_trend

        # تأكيد: ضغط شراء > 60% + حجم متصاعد
        if buy_pressure >= 0.65 and vol_trend == "increasing":
            result["confirmed"]   = True
            result["score_bonus"] = 12
            result["reason"]      = f"OrderFlow ✅ BP={buy_pressure:.0%} حجم↑"
        elif buy_pressure >= 0.55 and vol_trend != "decreasing":
            result["confirmed"]   = True
            result["score_bonus"] = 5
            result["reason"]      = f"OrderFlow ✓ BP={buy_pressure:.0%}"
        elif buy_pressure < 0.40:
            result["score_bonus"] = -8
            result["reason"]      = f"OrderFlow ⚠️ ضغط بيع BP={buy_pressure:.0%}"

    except Exception as e:
        logger.debug(f"[Q2] OrderFlow error: {e}")

    return result


# ══════════════════════════════════════════════════════
# [Q3] فلتر الوقت — تجنب آخر 25% من الشمعة
# ══════════════════════════════════════════════════════

def check_candle_timing(tf: str) -> Dict[str, Any]:
    """
    يتحقق من موضعنا داخل الشمعة الحالية.
    نتجنب الدخول في آخر 25% من عمر الشمعة
    لأن الشمعة قد تُغلق ضدنا.

    يرجع:
    {
        "ok": bool,
        "pct_elapsed": float,  ← نسبة الشمعة المنقضية
        "minutes_left": float,
        "reason": str
    }
    """
    result = {
        "ok":           True,
        "pct_elapsed":  0.5,
        "minutes_left": 0.0,
        "reason":       ""
    }

    try:
        tf_minutes = {
            "1m":1,"3m":3,"5m":5,"15m":15,"30m":30,
            "1h":60,"2h":120,"4h":240,"1d":1440
        }.get(tf.lower(), 15)

        now_ts    = time.time()
        bar_start = (now_ts // (tf_minutes * 60)) * (tf_minutes * 60)
        elapsed   = now_ts - bar_start
        pct       = elapsed / (tf_minutes * 60)
        mins_left = (tf_minutes * 60 - elapsed) / 60

        result["pct_elapsed"]  = round(pct, 3)
        result["minutes_left"] = round(mins_left, 1)

        if pct > (1.0 - Q_TIME_PCT_REMAIN):
            result["ok"]     = False
            result["reason"] = f"⏰ آخر {Q_TIME_PCT_REMAIN:.0%} من الشمعة ({mins_left:.0f}د متبقية)"
        else:
            result["reason"] = f"⏰ OK ({pct:.0%} منقضي، {mins_left:.0f}د متبقية)"

    except Exception as e:
        logger.debug(f"[Q3] Timing error: {e}")

    return result


# ══════════════════════════════════════════════════════
# [Q4] فلتر الارتباط — لا صفقتين متشابهتين
# ══════════════════════════════════════════════════════

# تصنيف العملات في مجموعات متشابهة
_CORRELATION_GROUPS = {
    "layer1":    {"BTC", "ETH", "SOL", "BNB", "ADA", "AVAX", "DOT", "ATOM", "NEAR", "APT", "SUI"},
    "defi":      {"AAVE", "UNI", "CRV", "COMP", "MKR", "SNX", "LDO"},
    "layer2":    {"ARB", "OP", "MATIC", "STRK", "ZK"},
    "meme":      {"DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI"},
    "oracle":    {"LINK", "BAND", "API3"},
    "exchange":  {"OKB", "BNB", "FTT"},
    "gaming":    {"AXS", "SAND", "MANA", "ENJ", "IMX"},
}

_OPEN_GROUPS: Dict[str, int] = {}  # group → count

def get_coin_group(symbol: str) -> Optional[str]:
    """يحدد مجموعة العملة"""
    coin = symbol.split("/")[0].split("#")[0].upper()
    for group, coins in _CORRELATION_GROUPS.items():
        if coin in coins:
            return group
    return None


def check_correlation(symbol: str, open_positions: List[str]) -> Dict[str, Any]:
    """
    يتحقق من عدم وجود صفقتين متشابهتين في نفس المجموعة.

    يرجع:
    {
        "ok": bool,
        "group": str,
        "group_count": int,
        "reason": str
    }
    """
    result = {
        "ok":          True,
        "group":       None,
        "group_count": 0,
        "reason":      ""
    }

    try:
        my_group = get_coin_group(symbol)
        if not my_group:
            result["reason"] = "مجموعة غير محددة — مسموح"
            return result

        result["group"] = my_group

        # عد الصفقات المفتوحة في نفس المجموعة
        count = sum(
            1 for pos_sym in open_positions
            if get_coin_group(pos_sym) == my_group
        )
        result["group_count"] = count

        if count >= Q_CORR_MAX_SYMBOLS:
            result["ok"]     = False
            result["reason"] = f"ارتباط عالٍ: {count} صفقة في مجموعة {my_group}"
        else:
            result["reason"] = f"ارتباط OK: {count}/{Q_CORR_MAX_SYMBOLS} في {my_group}"

    except Exception as e:
        logger.debug(f"[Q4] Correlation error: {e}")

    return result


# ══════════════════════════════════════════════════════
# [Q5] فلتر السيولة المتراكمة
# ══════════════════════════════════════════════════════

def check_liquidity_zones(df, current_price: float, atr: float) -> Dict[str, Any]:
    """
    يكتشف مناطق تراكم السيولة (أعلى/أدنى قمم/قيعان بارزة).
    يُحذر إذا السعر قريب من منطقة سيولة عالية (احتمال reversal).

    يرجع:
    {
        "near_resistance": bool,
        "near_support": bool,
        "resistance_level": float,
        "support_level": float,
        "score_bonus": int,
        "reason": str
    }
    """
    result = {
        "near_resistance": False,
        "near_support":    False,
        "resistance_level": 0.0,
        "support_level":    0.0,
        "score_bonus":      0,
        "reason":           ""
    }

    try:
        if len(df) < 30 or atr <= 0:
            return result

        seg    = df.iloc[-50:-1]
        highs  = seg["high"].tolist()
        lows   = seg["low"].tolist()

        # إيجاد مستويات السيولة (أعلى قمة وأدنى قاع)
        resistance = max(highs)
        support    = min(lows)

        result["resistance_level"] = round(resistance, 6)
        result["support_level"]    = round(support, 6)

        # قريب من مقاومة (في حدود 1 ATR)
        if (resistance - current_price) < atr * 1.0:
            result["near_resistance"] = True
            result["score_bonus"]     = -10
            result["reason"]          = f"⚠️ قريب من مقاومة {resistance:.4f}"

        # قريب من دعم (في حدود 0.5 ATR) = جيد للدخول
        elif (current_price - support) < atr * 2.0:
            result["near_support"]  = True
            result["score_bonus"]   = 5
            result["reason"]        = f"✅ فوق دعم {support:.4f}"
        else:
            result["reason"]        = "منطقة محايدة"

    except Exception as e:
        logger.debug(f"[Q5] Liquidity error: {e}")

    return result


# ══════════════════════════════════════════════════════
# الدالة الرئيسية — تجمع كل الفلاتر
# ══════════════════════════════════════════════════════

def run_quality_checks(
    symbol:         str,
    df,
    current_price:  float,
    atr:            float,
    tf:             str,
    open_positions: List[str],
    current_score:  int,
) -> Dict[str, Any]:
    """
    يشغّل كل فلاتر الجودة ويرجع:
    {
        "passed": bool,          ← هل اجتازت كل الفلاتر؟
        "final_score": int,      ← السكور بعد التعديل
        "block_reason": str,     ← سبب الرفض إن وُجد
        "details": dict,         ← تفاصيل كل فلتر
        "score_adjustments": int ← مجموع التعديلات
    }
    """
    details          = {}
    score_adj        = 0
    block_reason     = ""
    passed           = True

    # [Q1] Market Structure
    if Q_BOS_ENABLE:
        bos = check_market_structure(df)
        details["bos"] = bos
        score_adj += bos["score_bonus"]
        if bos["is_choch"]:
            block_reason = f"CHoCH: {bos['reason']}"
            passed       = False

    # [Q2] Order Flow
    if Q_ORDERFLOW_ENABLE and passed:
        of = check_order_flow(df)
        details["order_flow"] = of
        score_adj += of["score_bonus"]
        # لا نرفض على Order Flow وحده — فقط نعدّل السكور

    # [Q3] فلتر الوقت
    if Q_TIME_ENABLE and passed:
        tm = check_candle_timing(tf)
        details["timing"] = tm
        if not tm["ok"]:
            block_reason = tm["reason"]
            passed       = False

    # [Q4] فلتر الارتباط
    if Q_CORR_ENABLE and passed:
        corr = check_correlation(symbol, open_positions)
        details["correlation"] = corr
        if not corr["ok"]:
            block_reason = corr["reason"]
            passed       = False

    # [Q5] مناطق السيولة
    if Q_LIQUIDITY_ENABLE and passed:
        liq = check_liquidity_zones(df, current_price, atr)
        details["liquidity"] = liq
        score_adj += liq["score_bonus"]

    final_score = max(0, current_score + score_adj)

    return {
        "passed":           passed,
        "final_score":      final_score,
        "block_reason":     block_reason,
        "details":          details,
        "score_adjustments": score_adj,
    }
