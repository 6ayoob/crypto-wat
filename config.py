# -*- coding: utf-8 -*-
"""
config.py — نسخة محسّنة v2.0
التغييرات الرئيسية:
- HTF: 1h → 4h (تصفية أفضل للاتجاه)
- LTF: 5m → 15m (تقليل الضوضاء)
- SYMBOLS: مركّزة على أفضل 25 رمزاً سيولةً (بدل 300)
- variants مبسّطة: new + brt فقط (الأكثر فعالية)
- TRADE_BASE_USDT موحّد من مصدر واحد
- SCORE_THRESHOLD: 35 → 50
- MAX_OPEN_POSITIONS: 3 → 5
- DAILY_LOSS_LIMIT محسوب بنسبة من رأس المال
"""

from __future__ import annotations
import os, time, random
import requests
from typing import List, Optional

# ===============================
# 🔐 مفاتيح من البيئة فقط
# ===============================
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

OKX_API_KEY      = os.getenv("OKX_API_KEY", "")
OKX_API_SECRET   = os.getenv("OKX_API_SECRET", "")
OKX_PASSPHRASE   = os.getenv("OKX_PASSPHRASE", "")

# ===============================
# ⏱ إطارات زمنية — محسّنة
# ===============================
# HTF: 4h بدل 1h → تصفية أقوى للاتجاه العام
# LTF: 15m بدل 5m → تقليل الضوضاء ~60%
STRAT_HTF_TIMEFRAME = os.getenv("HTF_TIMEFRAME", "4h")
STRAT_LTF_TIMEFRAME = os.getenv("LTF_TIMEFRAME", "15m")

# للتوافق مع strategy.py
LTF_TIMEFRAME = STRAT_LTF_TIMEFRAME
HTF_TIMEFRAME = STRAT_HTF_TIMEFRAME

# ===============================
# 💰 حجم الصفقة — مصدر واحد موحّد
# ===============================
# TRADE_BASE_USDT هو المصدر الوحيد للحجم الأساسي
# لا يوجد TRADE_AMOUNT_USDT منفصل لتجنب التعارض
TRADE_BASE_USDT   = float(os.getenv("TRADE_BASE_USDT", "25.0"))
TRADE_AMOUNT_USDT = TRADE_BASE_USDT  # alias للتوافق مع imports قديمة

MIN_TRADE_USDT    = float(os.getenv("MIN_TRADE_USDT", "10.0"))
MAX_TRADE_USDT    = float(os.getenv("MAX_TRADE_USDT", "0.0"))   # 0 = غير مقيّد

# ===============================
# 📈 الرموز — 25 رمز مركّز بدل 300
# ===============================
# الرموز الأساسية: أعلى سيولة + أوضح حركة
SEED_SYMBOLS: List[str] = [
    # الماجورز (سيولة عالية + اتجاهات واضحة)
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",

    # ميد-كاب قوية
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "NEAR/USDT",
    "ATOM/USDT", "ARB/USDT", "OP/USDT",   "INJ/USDT", "APT/USDT",

    # DeFi
    "AAVE/USDT", "UNI/USDT",

    # ميمز (سيولة عالية فقط)
    "DOGE/USDT", "SHIB/USDT", "PEPE/USDT",

    # أخرى عالية السيولة
    "TRX/USDT", "LTC/USDT", "TON/USDT", "STX/USDT", "HBAR/USDT",
]

# ===============================
# ⚙️ إعدادات التوسع التلقائي
# ===============================
AUTO_EXPAND_SYMBOLS  = bool(int(os.getenv("AUTO_EXPAND_SYMBOLS", "1")))
TARGET_SYMBOLS_COUNT = int(os.getenv("TARGET_SYMBOLS_COUNT", "25"))   # بدل 60
MIN_USDT_VOL_24H     = float(os.getenv("MIN_USDT_VOL_24H", "15000000"))  # رُفع لـ 15M (جودة أعلى)
DEBUG_CONFIG_SYMBOLS = bool(int(os.getenv("DEBUG_CONFIG_SYMBOLS", "1")))

# ===============================
# 🎯 Variants — مبسّطة
# ===============================
# new: البولباك الكلاسيكي (الأساسي)
# brt: الاختراق فقط للرموز الأعلى سيولة
# تم حذف: old, srr, vbr (تعقيد بدون فائدة إضافية مثبتة)
ENABLE_BRT_TOP_N = int(os.getenv("ENABLE_BRT_TOP_N", "10"))  # أول 10 رموز فقط

# ===============================
# 📊 إعدادات التداول العامة — محسّنة
# ===============================
MAX_OPEN_POSITIONS    = int(os.getenv("MAX_OPEN_POSITIONS", "5"))     # رُفع من 3 إلى 5
FEE_BPS_ROUNDTRIP     = float(os.getenv("FEE_BPS_ROUNDTRIP", "16"))  # 0.08% * 2
MIN_NOTIONAL_USDT     = float(os.getenv("MIN_NOTIONAL_USDT", "5.0"))

# ===============================
# 🛡️ إدارة المخاطر — محسّنة
# ===============================
MAX_CONSEC_LOSSES     = int(os.getenv("MAX_CONSEC_LOSSES", "4"))      # رُفع من 3 إلى 4
MAX_TRADES_PER_DAY    = int(os.getenv("MAX_TRADES_PER_DAY", "15"))

# الخسارة اليومية: 8% من رأس المال المقدّر (أكثر واقعية من رقم ثابت)
_ESTIMATED_CAPITAL    = float(os.getenv("ESTIMATED_CAPITAL_USDT", "500.0"))
DAILY_LOSS_LIMIT_USDT = float(os.getenv(
    "DAILY_LOSS_LIMIT_USDT",
    str(round(_ESTIMATED_CAPITAL * 0.08, 1))   # 8% يومياً
))

# ===============================
# 📡 تيليجرام
# ===============================
STRAT_TG_SEND = bool(int(os.getenv("STRAT_TG_SEND", "1")))

# ===============================
# 🔢 السكور — عتبة أعلى للجودة
# ===============================
# رُفع من 35 إلى 50 لتصفية الإشارات الضعيفة
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "50"))

# ===============================
# ⚡️ جلب الرموز من OKX
# ===============================
_STABLE_BASES = {
    "USDT","USDC","DAI","FDUSD","TUSD","PYUSD",
    "EUR","TRY","BRL","AED","GBP","JPY"
}
_LEVERAGED_SUFFIXES = ("3L","3S","5L","5S","2L","2S","10L","10S")
OKX_BASE    = "https://www.okx.com"
TICKERS_URL = f"{OKX_BASE}/api/v5/market/tickers?instType=SPOT"
TIMEOUT_SEC = 12

def _normalize_symbol(s: str) -> str:
    return s.strip().upper().replace("-", "/")

def _dedupe_keep_order(seq) -> list:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

_REQ_SESSION: Optional[requests.Session] = None

def _get_session() -> requests.Session:
    global _REQ_SESSION
    if _REQ_SESSION is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "trading-bot/2.0 (okx liquidity filter)",
            "Accept": "application/json"
        })
        _REQ_SESSION = s
    return _REQ_SESSION

def _okx_get_json(url: str, attempts: int = 3):
    sess = _get_session()
    for a in range(attempts):
        try:
            r = sess.get(url, timeout=TIMEOUT_SEC)
            if r.status_code == 429:
                time.sleep((2 ** a) + random.random())
                continue
            r.raise_for_status()
            j = r.json()
            if str(j.get("code", "0")) not in ("0", "200"):
                time.sleep((2 ** a) + random.random())
                continue
            return j
        except Exception:
            time.sleep((2 ** a) + random.random())
    return None

def _fetch_okx_usdt_spot_ranked(min_usd_vol: float) -> List[tuple]:
    """يجلب الرموز مرتبة حسب حجم التداول 24h."""
    j = _okx_get_json(TICKERS_URL)
    if not j:
        return []
    rows = []
    for it in j.get("data", []):
        inst = (it.get("instId") or "").upper()
        if not inst.endswith("-USDT"):
            continue
        sym  = inst.replace("-", "/")
        base = sym.split("/", 1)[0].upper()

        # تصفية العملات المستقرة والرافعة المالية
        if base in _STABLE_BASES:
            continue
        if any(base.endswith(suf) for suf in _LEVERAGED_SUFFIXES):
            continue

        # حساب الحجم بالدولار
        vol = 0.0
        for key in ("volUsd", "volCcy24h", "vol24h"):
            v = it.get(key)
            if v:
                try:
                    vol = float(v)
                    if key == "vol24h":
                        last = float(it.get("last", 0) or 0)
                        vol  = vol * last
                    break
                except Exception:
                    pass

        if vol < min_usd_vol:
            continue

        rows.append((sym, vol))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

def _build_symbols_list(seed: List[str], target: int) -> List[str]:
    """
    يبني قائمة الرموز النهائية:
    1. يجلب الرموز المرتبة من OKX
    2. يُبقي الرموز الموجودة في SEED إن كانت بالقائمة
    3. يُكمّل بأعلى الرموز سيولة حتى target
    """
    base_list = _dedupe_keep_order(_normalize_symbol(s) for s in seed)

    if not AUTO_EXPAND_SYMBOLS:
        result = base_list[:target]
        if DEBUG_CONFIG_SYMBOLS:
            print(f"[config] AUTO_EXPAND off → {len(result)} symbols")
        return result

    ranked = _fetch_okx_usdt_spot_ranked(MIN_USDT_VOL_24H)
    if not ranked:
        if DEBUG_CONFIG_SYMBOLS:
            print("[config] ⚠️ OKX fetch failed → using SEED only")
        return base_list[:target]

    okx_set    = {s for s, _ in ranked}
    okx_ranked = [s for s, _ in ranked]

    # الأولوية للرموز الموجودة في SEED وعندها سيولة كافية
    kept   = [s for s in base_list if s in okx_set]
    extras = [s for s in okx_ranked if s not in set(kept)]
    result = _dedupe_keep_order(kept + extras)[:target]

    if DEBUG_CONFIG_SYMBOLS:
        print(f"[config] kept={len(kept)}, added={len(result)-len(kept)}, total={len(result)}")

    return result

# بناء قائمة الرموز الأساسية
try:
    _BASE_SYMBOLS = _build_symbols_list(SEED_SYMBOLS, TARGET_SYMBOLS_COUNT)
except Exception as e:
    print(f"[config] ⚠️ symbol build error: {e}")
    _BASE_SYMBOLS = [_normalize_symbol(s) for s in SEED_SYMBOLS[:TARGET_SYMBOLS_COUNT]]

# ===============================
# 🎯 توزيع الـ Variants — مبسّط
# ===============================
# كل رمز يحصل على variant واحد إضافي فقط (#brt) لأفضل 10
# بدل 4 variants لكل رمز (كان 300 → صار ~35)
_final_symbols = []
for idx, s in enumerate(_BASE_SYMBOLS):
    _final_symbols.append(s)                          # new (افتراضي)
    if idx < ENABLE_BRT_TOP_N:
        _final_symbols.append(f"{s}#brt")             # breakout للأعلى سيولة

SYMBOLS = _dedupe_keep_order(_final_symbols)

if DEBUG_CONFIG_SYMBOLS:
    print(f"[config] final SYMBOLS: {len(SYMBOLS)} | first 10: {SYMBOLS[:10]}")
    print(f"[config] HTF={STRAT_HTF_TIMEFRAME} | LTF={STRAT_LTF_TIMEFRAME}")
    print(f"[config] TRADE_BASE={TRADE_BASE_USDT}$ | SCORE_THR={SCORE_THRESHOLD}")
    print(f"[config] MAX_POS={MAX_OPEN_POSITIONS} | DAILY_LOSS_LIMIT={DAILY_LOSS_LIMIT_USDT}$")

# ===============================
# ✅ تصدير المتغيرات
# ===============================
__all__ = [
    # مفاتيح
    "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID",
    "OKX_API_KEY", "OKX_API_SECRET", "OKX_PASSPHRASE",
    # تايمفريمات
    "STRAT_HTF_TIMEFRAME", "STRAT_LTF_TIMEFRAME",
    "LTF_TIMEFRAME", "HTF_TIMEFRAME",
    # حجم الصفقة
    "TRADE_BASE_USDT", "TRADE_AMOUNT_USDT",
    "MIN_TRADE_USDT", "MAX_TRADE_USDT",
    # رموز
    "SYMBOLS", "SEED_SYMBOLS",
    # إعدادات عامة
    "MAX_OPEN_POSITIONS", "FEE_BPS_ROUNDTRIP", "MIN_NOTIONAL_USDT",
    # مخاطر
    "MAX_CONSEC_LOSSES", "MAX_TRADES_PER_DAY", "DAILY_LOSS_LIMIT_USDT",
    # تيليجرام
    "STRAT_TG_SEND",
    # سكور
    "SCORE_THRESHOLD",
]

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"SYMBOLS     : {len(SYMBOLS)}")
    print(f"HTF         : {STRAT_HTF_TIMEFRAME}")
    print(f"LTF         : {STRAT_LTF_TIMEFRAME}")
    print(f"TRADE_BASE  : {TRADE_BASE_USDT}$")
    print(f"MAX_POS     : {MAX_OPEN_POSITIONS}")
    print(f"SCORE_THR   : {SCORE_THRESHOLD}")
    print(f"DAILY_LIMIT : {DAILY_LOSS_LIMIT_USDT}$")
    print(f"First 10    : {SYMBOLS[:10]}")
    print(f"{'='*50}\n")
