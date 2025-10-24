# -*- coding: utf-8 -*-
"""
config.py — إعدادات موسّعة متوافقة مع strategy.py / main.py

الميزات:
- توسيع تلقائي لقائمة الرموز حسب سيولة OKX (24h USD volume).
- توزيع الاستراتيجيات (#old/#srr/#brt/#vbr) على أعلى الرموز سيولة.
- مفاتيح تيليغرام و OKX تُقرأ من البيئة فقط (لا مفاتيح صريحة في الكود).
- إطارات زمنية قابلة للتهيئة عبر ENV.
- طباعة تشخيصية مختصرة عن الرموز المتروكة والمضافة.
"""

from __future__ import annotations
import os, time, random, re
import requests
from typing import List, Tuple, Optional

# ===============================
# 🔐 مفاتيح من البيئة فقط (لا تضع مفاتيحك هنا)
# ===============================
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

OKX_API_KEY      = os.getenv("OKX_API_KEY", "")
OKX_API_SECRET   = os.getenv("OKX_API_SECRET", "")
OKX_PASSPHRASE   = os.getenv("OKX_PASSPHRASE", "")

# حجم الصفقة بالدولار (قابل للتعديل)
TRADE_BASE_USDT = 20.0
TRADE_AMOUNT_USDT = float(os.getenv("TRADE_AMOUNT_USDT", "45"))

# ===============================
# ⏱ إطارات زمنية تستخدمها الاستراتيجية
# ===============================
STRAT_HTF_TIMEFRAME = os.getenv("HTF_TIMEFRAME", "1h")  # إطار الاتجاه العام
STRAT_LTF_TIMEFRAME = os.getenv("LTF_TIMEFRAME", "5m")  # إطار الإشارات الدقيقة

# ===============================
# 📈 الرموز — قائمة بذور (سيتم توسيعها تلقائيًا حسب السيولة)
# ===============================
SEED_SYMBOLS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT",
    "TRX/USDT","TON/USDT","DOT/USDT","AVAX/USDT","LINK/USDT","LTC/USDT","BCH/USDT",
    "NEAR/USDT","ATOM/USDT","TIA/USDT","ARB/USDT","OP/USDT","INJ/USDT","STX/USDT",
    "PYTH/USDT","JTO/USDT","JUP/USDT","PEPE/USDT","FLOKI/USDT","SHIB/USDT","APT/USDT",
    "AAVE/USDT","UNI/USDT","GMX/USDT","DYDX/USDT"
]

# ===============================
# ⚙️ إعدادات التداول العامة
# ===============================
MAX_OPEN_POSITIONS  = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
FEE_BPS_ROUNDTRIP   = float(os.getenv("FEE_BPS_ROUNDTRIP", "16"))

# ===============================
# 📊 إعدادات توسيع الرموز والسيولة
# ===============================
AUTO_EXPAND_SYMBOLS  = bool(int(os.getenv("AUTO_EXPAND_SYMBOLS", "1")))
TARGET_SYMBOLS_COUNT = int(os.getenv("TARGET_SYMBOLS_COUNT", "60"))
MIN_USDT_VOL_24H     = float(os.getenv("MIN_USDT_VOL_24H", "5000000"))
DEBUG_CONFIG_SYMBOLS = bool(int(os.getenv("DEBUG_CONFIG_SYMBOLS", "1")))

# ===============================
# ⚡️ جلب الرموز من OKX
# ===============================
_STABLE_BASES = {"USDT","USDC","DAI","FDUSD","TUSD","PYUSD","EUR","TRY","BRL","AED","GBP","JPY"}
_LEVERAGED_SUFFIXES = ("3L","3S","5L","5S")
OKX_BASE = "https://www.okx.com"
TICKERS_URL = f"{OKX_BASE}/api/v5/market/tickers?instType=SPOT"
INSTR_URL   = f"{OKX_BASE}/api/v5/public/instruments?instType=SPOT"
TIMEOUT_SEC = 12

def _normalize_symbol(s): return s.strip().upper().replace("-", "/")
def _dedupe_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen: out.append(x); seen.add(x)
    return out

_REQ_SESSION: Optional[requests.Session] = None
def _get_session():
    global _REQ_SESSION
    if _REQ_SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "mk1-ai-bot/scan (okx liquidity filter)", "Accept": "application/json"})
        _REQ_SESSION = s
    return _REQ_SESSION

def _okx_get_json(url, attempts=3):
    sess = _get_session()
    for a in range(attempts):
        try:
            r = sess.get(url, timeout=TIMEOUT_SEC)
            if r.status_code == 429:
                time.sleep((2 ** a) + random.random()); continue
            r.raise_for_status()
            j = r.json()
            if str(j.get("code", "0")) not in ("0","200"):
                time.sleep((2 ** a) + random.random()); continue
            return j
        except Exception:
            time.sleep((2 ** a) + random.random())
    return None

def _fetch_okx_usdt_spot_ranked(min_usd_vol: float):
    j = _okx_get_json(TICKERS_URL)
    if not j: return []
    rows = []
    for it in j.get("data", []):
        inst = (it.get("instId") or "").upper()
        if not inst.endswith("-USDT"): continue
        vol = 0.0
        for key in ("volUsd", "volCcy24h", "vol24h"):
            v = it.get(key)
            if v:
                try:
                    vol = float(v)
                    if key == "vol24h":
                        last = float(it.get("last", 0) or 0)
                        vol = vol * last
                    break
                except Exception:
                    pass
        sym = inst.replace("-", "/")
        base = sym.split("/",1)[0].upper()
        if base in _STABLE_BASES or any(base.endswith(suf) for suf in _LEVERAGED_SUFFIXES):
            continue
        if vol < min_usd_vol: continue
        rows.append((sym, vol))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

def _expand_symbols_to_target(existing: list, target=60):
    base = _dedupe_keep_order(_normalize_symbol(s) for s in existing)
    ranked = _fetch_okx_usdt_spot_ranked(MIN_USDT_VOL_24H)
    if ranked:
        okx_ranked = [s for s,_ in ranked]
        kept = [s for s in base if s in okx_ranked]
        extras = [s for s in okx_ranked if s not in kept]
        out = (kept + extras)[:target]
        if DEBUG_CONFIG_SYMBOLS:
            print(f"[config] kept {len(kept)}, added {len(out)-len(kept)}")
        return out
    return base[:target]

try:
    SYMBOLS = _expand_symbols_to_target(SEED_SYMBOLS, TARGET_SYMBOLS_COUNT)
except Exception:
    SYMBOLS = list(SEED_SYMBOLS)[:TARGET_SYMBOLS_COUNT]

if DEBUG_CONFIG_SYMBOLS:
    print(f"[config] SYMBOLS ready: {len(SYMBOLS)} | first 10: {SYMBOLS[:10]}")

# ===============================
# 🎯 توزيع الاستراتيجيات على الرموز الأعلى
# ===============================
ENABLE_OLD_FOR_TOP_N = int(os.getenv("ENABLE_OLD_FOR_TOP_N", "10"))
ADD_SRR_TOP_N        = int(os.getenv("ADD_SRR_TOP_N", "40"))
ADD_BRT_TOP_N        = int(os.getenv("ADD_BRT_TOP_N", "10"))
ADD_VBR_TOP_N        = int(os.getenv("ADD_VBR_TOP_N", "40"))

_final_symbols = []
for idx, s in enumerate(SYMBOLS):
    _final_symbols.append(s)
    if idx < ENABLE_OLD_FOR_TOP_N: _final_symbols.append(f"{s}#old")
    if idx < ADD_SRR_TOP_N:        _final_symbols.append(f"{s}#srr")
    if idx < ADD_BRT_TOP_N:        _final_symbols.append(f"{s}#brt")
    if idx < ADD_VBR_TOP_N:        _final_symbols.append(f"{s}#vbr")

SYMBOLS = _dedupe_keep_order(_final_symbols)
if DEBUG_CONFIG_SYMBOLS:
    print(f"[config] final SYMBOLS: {len(SYMBOLS)} | first 12: {SYMBOLS[:12]}")

# ===============================
# 🧩 تكامل كامل مع strategy.py
# ===============================
LTF_TIMEFRAME = STRAT_LTF_TIMEFRAME
HTF_TIMEFRAME = STRAT_HTF_TIMEFRAME
MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", "5.0"))
MAX_CONSEC_LOSSES = int(os.getenv("MAX_CONSEC_LOSSES", "3"))
DAILY_LOSS_LIMIT_USDT = float(os.getenv("DAILY_LOSS_LIMIT_USDT", "50.0"))
STRAT_TG_SEND = bool(int(os.getenv("STRAT_TG_SEND", "1")))

# ===============================
# ✅ تصدير المتغيرات
# ===============================
__all__ = [
    "TELEGRAM_TOKEN","TELEGRAM_CHAT_ID",
    "OKX_API_KEY","OKX_API_SECRET","OKX_PASSPHRASE",
    "STRAT_HTF_TIMEFRAME","STRAT_LTF_TIMEFRAME",
    "LTF_TIMEFRAME","HTF_TIMEFRAME",
    "TRADE_BASE_USDT","TRADE_AMOUNT_USDT","MAX_OPEN_POSITIONS",
    "FEE_BPS_ROUNDTRIP","MIN_NOTIONAL_USDT",
    "MAX_CONSEC_LOSSES","DAILY_LOSS_LIMIT_USDT",
    "STRAT_TG_SEND","SYMBOLS"
]

if __name__ == "__main__":
    print(f"[config] SYMBOLS ready: {len(SYMBOLS)} | first 12: {SYMBOLS[:12]}")
