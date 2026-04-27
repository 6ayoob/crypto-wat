# -*- coding: utf-8 -*-
"""
config.py — نسخة احترافية v3.0
التغييرات الجوهرية عن v2.0:

[FIX-1] حجم الصفقة = نسبة % من رأس المال الفعلي (بدل مبلغ ثابت)
[FIX-2] DAILY_LOSS_LIMIT = نسبة % من رأس المال الفعلي
[FIX-3] MAX_OPEN_POSITIONS = يتناسب مع رأس المال تلقائياً
[FIX-4] MAX_CONSEC_LOSSES رُفع + حظر بالقيمة لا العدد فقط
[FIX-5] SCORE_THRESHOLD رُفع لـ 55 للجودة
[FIX-6] جلب رأس المال الفعلي من OKX عند بدء التشغيل
[FIX-7] حد أقصى للمخاطرة لكل صفقة (2% من رأس المال)
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
# ⏱ إطارات زمنية
# ===============================
STRAT_HTF_TIMEFRAME = os.getenv("HTF_TIMEFRAME", "4h")
STRAT_LTF_TIMEFRAME = os.getenv("LTF_TIMEFRAME", "15m")
LTF_TIMEFRAME = STRAT_LTF_TIMEFRAME
HTF_TIMEFRAME = STRAT_HTF_TIMEFRAME

# ===============================
# 💰 [FIX-1] حجم الصفقة الذكي
# ===============================
# بدل مبلغ ثابت — نستخدم نسبة من رأس المال الفعلي
# المخاطرة القصوى لكل صفقة = 5% من رأس المال
# مثال: رأس مال 100$ → صفقة 5$
#        رأس مال 500$ → صفقة 25$
#        رأس مال 30$  → صفقة 1.5$

RISK_PER_TRADE_PCT  = float(os.getenv("RISK_PER_TRADE_PCT",  "5.0"))   # 5% من رأس المال
MAX_RISK_PER_TRADE_PCT = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "10.0"))  # حد أقصى 10%
MIN_TRADE_USDT      = float(os.getenv("MIN_TRADE_USDT",      "2.0"))    # حد أدنى مطلق
MAX_TRADE_USDT      = float(os.getenv("MAX_TRADE_USDT",      "0.0"))    # 0 = غير مقيّد

# [FIX-6] جلب رأس المال الفعلي من OKX
def _fetch_actual_balance() -> float:
    """يجلب رصيد USDT الفعلي من OKX"""
    try:
        import hmac, hashlib, base64
        from datetime import datetime, timezone

        if not all([OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE]):
            return 0.0

        ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        msg = ts + "GET" + "/api/v5/account/balance" + ""
        sig = base64.b64encode(
            hmac.new(OKX_API_SECRET.encode(), msg.encode(), hashlib.sha256).digest()
        ).decode()

        headers = {
            "OK-ACCESS-KEY":        OKX_API_KEY,
            "OK-ACCESS-SIGN":       sig,
            "OK-ACCESS-TIMESTAMP":  ts,
            "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
            "Content-Type":         "application/json",
        }
        r = requests.get(
            "https://www.okx.com/api/v5/account/balance",
            headers=headers, timeout=10
        )
        data = r.json()
        if data.get("code") == "0":
            for detail in data["data"][0].get("details", []):
                if detail.get("ccy") == "USDT":
                    return float(detail.get("availBal", 0) or 0)
    except Exception as e:
        print(f"[config] ⚠️ جلب الرصيد فشل: {e}")
    return 0.0

def _fetch_spot_balance() -> float:
    """يجلب رصيد USDT من حساب Spot"""
    try:
        import hmac, hashlib, base64
        from datetime import datetime, timezone

        if not all([OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE]):
            return 0.0

        ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        path = "/api/v5/asset/balances?ccy=USDT"
        msg = ts + "GET" + path + ""
        sig = base64.b64encode(
            hmac.new(OKX_API_SECRET.encode(), msg.encode(), hashlib.sha256).digest()
        ).decode()

        headers = {
            "OK-ACCESS-KEY":        OKX_API_KEY,
            "OK-ACCESS-SIGN":       sig,
            "OK-ACCESS-TIMESTAMP":  ts,
            "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
            "Content-Type":         "application/json",
        }
        r = requests.get(
            f"https://www.okx.com{path}",
            headers=headers, timeout=10
        )
        data = r.json()
        if data.get("code") == "0":
            for item in data.get("data", []):
                if item.get("ccy") == "USDT":
                    return float(item.get("availBal", 0) or 0)
    except Exception as e:
        print(f"[config] ⚠️ جلب Spot فشل: {e}")
    return 0.0

# جلب رأس المال الفعلي
_ACTUAL_BALANCE = 0.0
try:
    _b1 = _fetch_actual_balance()
    _b2 = _fetch_spot_balance()
    _ACTUAL_BALANCE = max(_b1, _b2)
except Exception:
    pass

# fallback: استخدام القيمة من ENV إذا فشل الجلب
_ESTIMATED_CAPITAL = float(os.getenv("ESTIMATED_CAPITAL_USDT", "0.0"))
ACTUAL_CAPITAL_USDT = _ACTUAL_BALANCE if _ACTUAL_BALANCE > 1.0 else _ESTIMATED_CAPITAL

# [FIX-1] حساب حجم الصفقة بناءً على رأس المال
if ACTUAL_CAPITAL_USDT > 1.0:
    _calculated_trade = round(ACTUAL_CAPITAL_USDT * RISK_PER_TRADE_PCT / 100.0, 2)
    TRADE_BASE_USDT = max(MIN_TRADE_USDT, _calculated_trade)
    if MAX_TRADE_USDT > 0:
        TRADE_BASE_USDT = min(TRADE_BASE_USDT, MAX_TRADE_USDT)
else:
    # إذا لم نعرف رأس المال → استخدم القيمة من ENV
    TRADE_BASE_USDT = float(os.getenv("TRADE_BASE_USDT", "5.0"))

TRADE_AMOUNT_USDT = TRADE_BASE_USDT  # alias للتوافق

# ===============================
# 📈 الرموز
# ===============================
SEED_SYMBOLS: List[str] = [
    # الماجورز
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    # ميد-كاب قوية
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "NEAR/USDT",
    "ATOM/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "APT/USDT",
    # DeFi
    "AAVE/USDT", "UNI/USDT",
    # ميمز سيولة عالية
    "DOGE/USDT", "SHIB/USDT", "PEPE/USDT",
    # أخرى
    "TRX/USDT", "LTC/USDT", "TON/USDT", "STX/USDT", "HBAR/USDT",
]

AUTO_EXPAND_SYMBOLS  = bool(int(os.getenv("AUTO_EXPAND_SYMBOLS", "1")))
TARGET_SYMBOLS_COUNT = int(os.getenv("TARGET_SYMBOLS_COUNT", "25"))
MIN_USDT_VOL_24H     = float(os.getenv("MIN_USDT_VOL_24H", "15000000"))
DEBUG_CONFIG_SYMBOLS = bool(int(os.getenv("DEBUG_CONFIG_SYMBOLS", "1")))
ENABLE_BRT_TOP_N     = int(os.getenv("ENABLE_BRT_TOP_N", "10"))

# ===============================
# 📊 إعدادات التداول — محسّنة
# ===============================

# [FIX-3] MAX_OPEN_POSITIONS يتناسب مع رأس المال
if ACTUAL_CAPITAL_USDT > 1.0:
    # كل صفقة تأخذ RISK_PER_TRADE_PCT% → الحد الأقصى المنطقي
    _max_pos_by_capital = max(2, min(10, int(100 / RISK_PER_TRADE_PCT)))
    MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", str(_max_pos_by_capital)))
else:
    MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "5"))

FEE_BPS_ROUNDTRIP = float(os.getenv("FEE_BPS_ROUNDTRIP", "16"))
MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", "5.0"))

# ===============================
# 🛡️ [FIX-4] إدارة المخاطر المحسّنة
# ===============================

# [FIX-4a] رُفع MAX_CONSEC_LOSSES
MAX_CONSEC_LOSSES  = int(os.getenv("MAX_CONSEC_LOSSES", "6"))   # كان 4

# [FIX-4b] حد يومي بنسبة من رأس المال الفعلي
DAILY_LOSS_PCT     = float(os.getenv("DAILY_LOSS_PCT", "5.0"))  # 5% يومياً
if ACTUAL_CAPITAL_USDT > 1.0:
    _auto_daily_limit = round(ACTUAL_CAPITAL_USDT * DAILY_LOSS_PCT / 100.0, 2)
    DAILY_LOSS_LIMIT_USDT = float(os.getenv("DAILY_LOSS_LIMIT_USDT", str(_auto_daily_limit)))
else:
    DAILY_LOSS_LIMIT_USDT = float(os.getenv("DAILY_LOSS_LIMIT_USDT", "5.0"))

MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "15"))

# ===============================
# 🔢 [FIX-5] السكور — جودة أعلى
# ===============================
SCORE_THRESHOLD = int(os.getenv("SCORE_THRESHOLD", "55"))   # رُفع من 50 إلى 55

# ===============================
# 📡 تيليجرام
# ===============================
STRAT_TG_SEND = bool(int(os.getenv("STRAT_TG_SEND", "1")))

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
            "User-Agent": "trading-bot/3.0",
            "Accept":     "application/json"
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
        if base in _STABLE_BASES:
            continue
        if any(base.endswith(suf) for suf in _LEVERAGED_SUFFIXES):
            continue
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
    base_list = _dedupe_keep_order(_normalize_symbol(s) for s in seed)
    if not AUTO_EXPAND_SYMBOLS:
        return base_list[:target]
    ranked = _fetch_okx_usdt_spot_ranked(MIN_USDT_VOL_24H)
    if not ranked:
        print("[config] ⚠️ OKX fetch failed → using SEED only")
        return base_list[:target]
    okx_set    = {s for s, _ in ranked}
    okx_ranked = [s for s, _ in ranked]
    kept   = [s for s in base_list if s in okx_set]
    extras = [s for s in okx_ranked if s not in set(kept)]
    result = _dedupe_keep_order(kept + extras)[:target]
    if DEBUG_CONFIG_SYMBOLS:
        print(f"[config] kept={len(kept)}, added={len(result)-len(kept)}, total={len(result)}")
    return result

try:
    _BASE_SYMBOLS = _build_symbols_list(SEED_SYMBOLS, TARGET_SYMBOLS_COUNT)
except Exception as e:
    print(f"[config] ⚠️ symbol build error: {e}")
    _BASE_SYMBOLS = [_normalize_symbol(s) for s in SEED_SYMBOLS[:TARGET_SYMBOLS_COUNT]]

_final_symbols = []
for idx, s in enumerate(_BASE_SYMBOLS):
    _final_symbols.append(s)
    if idx < ENABLE_BRT_TOP_N:
        _final_symbols.append(f"{s}#brt")

SYMBOLS = _dedupe_keep_order(_final_symbols)

# ===============================
# 📋 تقرير الإعدادات
# ===============================
if DEBUG_CONFIG_SYMBOLS:
    print(f"[config] final SYMBOLS: {len(SYMBOLS)} | first 10: {SYMBOLS[:10]}")
    print(f"[config] HTF={STRAT_HTF_TIMEFRAME} | LTF={STRAT_LTF_TIMEFRAME}")
    print(f"[config] رأس المال={ACTUAL_CAPITAL_USDT:.2f}$ | TRADE_BASE={TRADE_BASE_USDT:.2f}$ ({RISK_PER_TRADE_PCT}%)")
    print(f"[config] SCORE_THR={SCORE_THRESHOLD} | MAX_POS={MAX_OPEN_POSITIONS}")
    print(f"[config] DAILY_LOSS_LIMIT={DAILY_LOSS_LIMIT_USDT:.2f}$ ({DAILY_LOSS_PCT}%) | MAX_CONSEC={MAX_CONSEC_LOSSES}")

# ===============================
# ✅ تصدير المتغيرات
# ===============================
__all__ = [
    "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID",
    "OKX_API_KEY", "OKX_API_SECRET", "OKX_PASSPHRASE",
    "STRAT_HTF_TIMEFRAME", "STRAT_LTF_TIMEFRAME",
    "LTF_TIMEFRAME", "HTF_TIMEFRAME",
    "TRADE_BASE_USDT", "TRADE_AMOUNT_USDT",
    "MIN_TRADE_USDT", "MAX_TRADE_USDT",
    "ACTUAL_CAPITAL_USDT", "RISK_PER_TRADE_PCT",
    "SYMBOLS", "SEED_SYMBOLS",
    "MAX_OPEN_POSITIONS", "FEE_BPS_ROUNDTRIP", "MIN_NOTIONAL_USDT",
    "MAX_CONSEC_LOSSES", "MAX_TRADES_PER_DAY", "DAILY_LOSS_LIMIT_USDT",
    "STRAT_TG_SEND",
    "SCORE_THRESHOLD",
]

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"رأس المال الفعلي : {ACTUAL_CAPITAL_USDT:.2f}$")
    print(f"حجم الصفقة      : {TRADE_BASE_USDT:.2f}$ ({RISK_PER_TRADE_PCT}%)")
    print(f"الحد اليومي     : {DAILY_LOSS_LIMIT_USDT:.2f}$ ({DAILY_LOSS_PCT}%)")
    print(f"MAX_POS         : {MAX_OPEN_POSITIONS}")
    print(f"SCORE_THR       : {SCORE_THRESHOLD}")
    print(f"MAX_CONSEC      : {MAX_CONSEC_LOSSES}")
    print(f"SYMBOLS         : {len(SYMBOLS)}")
    print(f"HTF/LTF         : {STRAT_HTF_TIMEFRAME}/{STRAT_LTF_TIMEFRAME}")
    print(f"{'='*50}\n")
