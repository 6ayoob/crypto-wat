# -*- coding: utf-8 -*-
"""
config.py — إعدادات موسّعة متوافقة مع strategy.py / main.py
- توسيع تلقائي لقائمة الرموز حسب سيولة OKX
- توزيع الاستراتيجيات (#new/#old/#srr/#brt/#vbr) على أعلى الرموز سيولة
- مفاتيح تيليغرام و OKX تُقرأ من متغيرات البيئة (لا مفاتيح صريحة في الكود)
- إطارات زمنية قابلة للتهيئة وتُقرأ من البيئة

التعديلات الافتراضية (يمكن تغييرها عبر ENV):
- STRAT_HTF_TIMEFRAME = "1h"  (كان 15m)
- TARGET_SYMBOLS_COUNT = 60   (كان 100)
- MIN_USDT_VOL_24H = 5_000_000$  (كان 1_000_000$)
- توزيع الاستراتيجيات: old=10, srr=40, brt=10, vbr=40
"""

import os, time, random, re
import requests
from typing import List, Tuple, Optional

# ===============================
# 🔐 مفاتيح من البيئة فقط (لا تضع مفاتيحك هنا)
# ===============================
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# مفاتيح OKX (مجرّد تعريفات — okx_api يقرأ من البيئة أيضًا)
OKX_API_KEY      = os.getenv("OKX_API_KEY", "")
OKX_API_SECRET   = os.getenv("OKX_API_SECRET", "")
OKX_PASSPHRASE   = os.getenv("OKX_PASSPHRASE", "")

# ===============================
# ⏱ إطارات زمنية تستخدمها الاستراتيجية
# ===============================
STRAT_HTF_TIMEFRAME = os.getenv("HTF_TIMEFRAME", "1h")  # إطار السياق (HTF) — أنعم وأقل رفض HTF
STRAT_LTF_TIMEFRAME = os.getenv("LTF_TIMEFRAME", "5m")  # إطار التنفيذ (LTF)

# ===============================
# 📈 الرموز — قائمة بذور (سيتم فلترتها/تكميلها تلقائيًا)
# ملاحظة: يوجد dedupe لاحقًا، فلا تقلق من أي تكرارات عرضية هنا.
# ===============================
SEED_SYMBOLS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT","TRX/USDT","TON/USDT","DOT/USDT",
    "AVAX/USDT","LINK/USDT","LTC/USDT","BCH/USDT","ETC/USDT","NEAR/USDT","ATOM/USDT","TIA/USDT","ARB/USDT",
    "OP/USDT","STRK/USDT","ZK/USDT","SUI/USDT","APT/USDT","INJ/USDT","STX/USDT","PYTH/USDT","JTO/USDT","JUP/USDT",
    "WIF/USDT","BONK/USDT","PEPE/USDT","FLOKI/USDT","SHIB/USDT","MEME/USDT","KAIA/USDT","AAVE/USDT","UNI/USDT",
    "SUSHI/USDT","COMP/USDT","SNX/USDT","LDO/USDT","CRV/USDT","BAL/USDT","YFI/USDT","GMX/USDT","DYDX/USDT",
    "1INCH/USDT","CVX/USDT","RPL/USDT","FXS/USDT","SSV/USDT","LQTY/USDT","APE/USDT","ENS/USDT","GRT/USDT","CHZ/USDT",
    "AXS/USDT","SAND/USDT","MANA/USDT","IMX/USDT","GALA/USDT","RON/USDT","MAGIC/USDT","MINA/USDT","ICP/USDT",
    "FIL/USDT","AR/USDT","STORJ/USDT","SC/USDT","HBAR/USDT","EGLD/USDT","ALGO/USDT","THETA/USDT","CFX/USDT",
    "XTZ/USDT","ZIL/USDT","NEO/USDT","QTUM/USDT","IOTA/USDT","ONDO/USDT","ETHFI/USDT","PENDLE/USDT","AEVO/USDT",
    "ZRO/USDT","BOME/USDT","ORDI/USDT","SATS/USDT","CELO/USDT","W/USDT","SLERF/USDT","RAY/USDT","S/USDT",
    "PRCL/USDT","GHST/USDT","OKB/USDT","PEOPLE/USDT","IP/USDT","ELF/USDT","SKL/USDT","COTI/USDT","ID/USDT",
    "EOS/USDT","ETHW/USDT","FLOW/USDT","GAL/USDT","GLMR/USDT","GMT/USDT","IOST/USDT","JASMY/USDT","JOE/USDT",
    "KLAY/USDT","KNC/USDT","KSM/USDT","LINA/USDT","LPT/USDT","LRC/USDT","MASK/USDT","MAV/USDT","NOT/USDT","OM/USDT",
    "ONT/USDT","POLS/USDT","PRIME/USDT","RSR/USDT","SFP/USDT","TRB/USDT","UMA/USDT","UNFI/USDT",
    "USDC/USDT","WLD/USDT","WOO/USDT","XAI/USDT","XLM/USDT","YGG/USDT","ZETA/USDT","ZRX/USDT","BLUR/USDT","BNT/USDT",
    "BICO/USDT","CELR/USDT","CFG/USDT","CYBER/USDT","DODO/USDT","DYM/USDT","EDU/USDT","EUL/USDT","FIDA/USDT","FLM/USDT",
    "FRONT/USDT","FX/USDT","LAYER/USDT","HIGH/USDT","HOOK/USDT","HOT/USDT","IDEX/USDT","ILV/USDT","LOKA/USDT","MBOX/USDT",
    "NKN/USDT","NMR/USDT","OL/USDT","PERP/USDT","PHA/USDT","PLA/USDT","RLC/USDT","STMX/USDT","STPT/USDT","SUPER/USDT",
    "SWEAT/USDT","SYS/USDT","TLM/USDT"
]

# ===============================
# ⏱ إعدادات التداول العامة
# ===============================
TRADE_AMOUNT_USDT   = float(os.getenv("TRADE_AMOUNT_USDT", "45"))
MAX_OPEN_POSITIONS  = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

# ===============================
# 🧮 الرسوم (round-trip) بالـ bps
# ===============================
FEE_BPS_ROUNDTRIP = float(os.getenv("FEE_BPS_ROUNDTRIP", "16"))

# ===============================
# ⚙️ خيارات التوسيع التلقائي + الطباعة
# ===============================
AUTO_EXPAND_SYMBOLS     = bool(int(os.getenv("AUTO_EXPAND_SYMBOLS", "1")))
TARGET_SYMBOLS_COUNT    = int(os.getenv("TARGET_SYMBOLS_COUNT", "60"))   # ← كان 100
DEBUG_CONFIG_SYMBOLS    = bool(int(os.getenv("DEBUG_CONFIG_SYMBOLS", "1")))  # اطبع موجزًا عن القائمة
PRINT_SYMBOLS_ON_IMPORT = bool(int(os.getenv("PRINT_SYMBOLS_ON_IMPORT", "0")))  # اطبع القائمة كاملة عند الاستيراد

# === فلترة/سيولة قابلة للتهيئة ===
ALLOWED_QUOTE    = os.getenv("ALLOWED_QUOTE", "USDT").upper()
MIN_USDT_VOL_24H = float(os.getenv("MIN_USDT_VOL_24H", "5000000"))  # ← كان 1M

# ✅ استبعاد يعتمد على BASE فقط
EXCLUDE_BASE_REGEX = os.getenv("EXCLUDE_BASE_REGEX", r"(TEST|IOU)")
INCLUDE_REGEX      = os.getenv("INCLUDE_REGEX", "")

_STABLE_BASES        = {"USDT","USDC","DAI","FDUSD","TUSD","PYUSD","EUR","TRY","BRL","AED","GBP","JPY"}
_LEVERAGED_SUFFIXES  = ("3L","3S","5L","5S")

OKX_TICKERS_CACHE_SEC = int(os.getenv("OKX_TICKERS_CACHE_SEC", "90"))
OKX_BASE  = "https://www.okx.com"
TICKERS_URL = f"{OKX_BASE}/api/v5/market/tickers?instType=SPOT"
INSTR_URL   = f"{OKX_BASE}/api/v5/public/instruments?instType=SPOT"
TIMEOUT_SEC = 12

# ============ أدوات مساعدة ============
def _normalize_symbol(s: str) -> str:
    return s.strip().upper().replace("-", "/")

def _dedupe_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

_REQ_SESSION: Optional[requests.Session] = None
def _get_session() -> requests.Session:
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

def _fetch_okx_usdt_spot_ranked(min_usd_vol: float) -> List[Tuple[str, float]]:
    j = _okx_get_json(TICKERS_URL, attempts=3)
    if not j: return []
    rows: List[Tuple[str, float]] = []
    for it in j.get("data", []):
        inst = (it.get("instId") or "").upper()
        if not inst.endswith(f"-{ALLOWED_QUOTE}"):
            continue
        # تقدير حجم USD
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
        try:
            sym = inst.replace("-", "/")
            base = sym.split("/",1)[0].upper()
        except Exception:
            continue
        if base in _STABLE_BASES or any(base.endswith(suf) for suf in _LEVERAGED_SUFFIXES):
            continue
        if EXCLUDE_BASE_REGEX and re.search(EXCLUDE_BASE_REGEX, base, re.IGNORECASE):
            continue
        if vol < min_usd_vol:
            continue
        rows.append((sym, vol))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

def _fetch_okx_spot_supported() -> List[str]:
    j = _okx_get_json(INSTR_URL, attempts=2)
    if not j: return []
    out = []
    for it in j.get("data", []):
        inst = (it.get("instId") or "").upper()
        if not inst or f"-{ALLOWED_QUOTE}" not in inst:
            continue
        sym = inst.replace("-", "/")
        base = sym.split("/",1)[0].upper()
        if base in _STABLE_BASES or any(base.endswith(suf) for suf in _LEVERAGED_SUFFIXES):
            continue
        if EXCLUDE_BASE_REGEX and re.search(EXCLUDE_BASE_REGEX, base, re.IGNORECASE):
            continue
        out.append(sym)
    return out

def _expand_symbols_to_target(existing: List[str], target=100) -> List[str]:
    base = _dedupe_keep_order(_normalize_symbol(s) for s in existing)
    # إن وُجد regex للإدراج الإجباري
    if INCLUDE_REGEX:
        base = [s for s in base if (re.search(INCLUDE_REGEX, s.split("/")[0], re.IGNORECASE) or True)]
    ranked = _fetch_okx_usdt_spot_ranked(MIN_USDT_VOL_24H)
    if ranked:
        okx_ranked = [s for s,_ in ranked]
        okx_set = set(okx_ranked)
        kept = [s for s in base if (s in okx_set)]
        extras = [s for s in okx_ranked if s not in kept]
        out = (kept + extras)[:target]
        if DEBUG_CONFIG_SYMBOLS:
            missing = [s for s in base if s not in okx_set]
            print(f"[config] kept {len(kept)}, added {len(out)-len(kept)}, missing_or_filtered: {missing[:10]}")
        return out
    supported = set(_fetch_okx_spot_supported())
    if supported:
        kept = [s for s in base if s in supported]
        extras = [s for s in supported if s not in kept]
        out = (kept + extras)[:target]
        if DEBUG_CONFIG_SYMBOLS:
            print(f"[config] tickers failed; used instruments. kept={len(kept)} total={len(out)}")
        return out
    out = [s for s in base][:target]
    if DEBUG_CONFIG_SYMBOLS:
        print(f"[config] OKX fetch failed; using existing ({len(out)})")
    return out

# ============ التنفيذ: بناء SYMBOLS ============
try:
    if bool(AUTO_EXPAND_SYMBOLS):
        SYMBOLS = _expand_symbols_to_target(SEED_SYMBOLS, TARGET_SYMBOLS_COUNT)
    else:
        SYMBOLS = [
            s for s in _dedupe_keep_order(_normalize_symbol(s) for s in SEED_SYMBOLS)
        ][:TARGET_SYMBOLS_COUNT]
except Exception:
    SYMBOLS = list(_dedupe_keep_order(_normalize_symbol(s) for s in SEED_SYMBOLS))[:TARGET_SYMBOLS_COUNT]

if DEBUG_CONFIG_SYMBOLS:
    print(f"[config] SYMBOLS ready: {len(SYMBOLS)} | first 10: {SYMBOLS[:10]}")

# ===============================
# 🎯 توزيع الاستراتيجيات على أعلى الرموز سيولة
# (يضيف نسخ #old/#srr/#brt/#vbr لأعلى N رموز — والباقي يظل #new)
# ===============================
ENABLE_OLD_FOR_TOP_N = int(os.getenv("ENABLE_OLD_FOR_TOP_N", "10"))  # ← كان 20
ADD_SRR_TOP_N        = int(os.getenv("ADD_SRR_TOP_N", "40"))         # ← كان 15
ADD_BRT_TOP_N        = int(os.getenv("ADD_BRT_TOP_N", "10"))         # ← كان 15
ADD_VBR_TOP_N        = int(os.getenv("ADD_VBR_TOP_N", "40"))         # ← كان 15

# خيار اختياري لاستخدام استراتيجية SR/RSI لاحقًا (لو فعّلتها في strategy.py)
ADD_SR_TOP_N         = int(os.getenv("ADD_SR_TOP_N", "0"))  # افتراضيًا 0 حتى لا تُنشئ #sr بدون دعم

_final_symbols: List[str] = []
for idx, s in enumerate(SYMBOLS):
    _final_symbols.append(s)                 # الافتراضي (#new)
    if idx < ENABLE_OLD_FOR_TOP_N: _final_symbols.append(f"{s}#old")
    if idx < ADD_SRR_TOP_N:        _final_symbols.append(f"{s}#srr")
    if idx < ADD_BRT_TOP_N:        _final_symbols.append(f"{s}#brt")
    if idx < ADD_VBR_TOP_N:        _final_symbols.append(f"{s}#vbr")
    if idx < ADD_SR_TOP_N:         _final_symbols.append(f"{s}#sr")   # يعمل فقط لو strategy تدعمها

# إزالة أي تكرارات مع الحفاظ على الترتيب
def _dedupe_strats(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

SYMBOLS = _dedupe_strats(_final_symbols)

if DEBUG_CONFIG_SYMBOLS:
    print(f"[config] final SYMBOLS: {len(SYMBOLS)} | first 12: {SYMBOLS[:12]}")

# طباعة اختيارية عند الاستيراد (لو تبغى ترى القائمة بالكامل فور التشغيل)
if PRINT_SYMBOLS_ON_IMPORT:
    try:
        print("----- SYMBOLS (all) -----")
        for s in SYMBOLS:
            print(s)
        print("----- /SYMBOLS -----")
    except Exception:
        pass

# تشغيل كملف مستقل: طباعة موجزة + كاملة (لتشخيص سريع)
if __name__ == "__main__":
    print(f"[config] SYMBOLS ready: {len(SYMBOLS)} | first 12: {SYMBOLS[:12]}")
    show_all = os.getenv("SHOW_ALL_SYMBOLS", "0").lower() in ("1","true","yes","y")
    if show_all:
        for s in SYMBOLS:
            print(s)
