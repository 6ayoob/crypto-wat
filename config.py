
# ===============================
# 🔑 مفاتيح API لـ OKX (استخدم متغيرات بيئة للأمان)
# ===============================
import os

API_KEY = "6e2d2b3f-636a-424a-a97e-5154e39e525a"
SECRET_KEY = "D4B9966385BEE5A7B7D8791BA5C0539F"
PASSPHRASE = "Ta123456&"
# ===============================
# 🤖 Telegram
# ===============================


TELEGRAM_TOKEN = "8300868885:AAEx8Zxdkz9CRUHmjJ0vvn6L3kC2kOPCHuk"
TELEGRAM_CHAT_ID = "658712542"
# ===============================
# 📈 الرموز — قائمة منقّحة (سيتم فلترتها وإكمالها تلقائياً من OKX عند الإقلاع)
# ملاحظة: لا تعتمد على هذه القائمة فقط؛ سيتم التحقق من دعم OKX وإكمالها حتى 100.
# ===============================

import os, time, random
import requests

# قائمتك الأصلية (ستُطبّع للأحرف الكبيرة وتزال التكرارات)
SYMBOLS = [
    # DeFi / Bluechips
    "AAVE/USDT", "UNI/USDT", "SUSHI/USDT", "COMP/USDT", "MKR/USDT",
    "SNX/USDT", "LDO/USDT", "GRT/USDT", "LINK/USDT",

    # Layer 1 / Majors
    "ETH/USDT", "SOL/USDT", "ADA/USDT", "AVAX/USDT", "NEAR/USDT",
    "ATOM/USDT", "DOT/USDT",

    # Gaming/Metaverse
    "MANA/USDT", "AXS/USDT", "BIGTIME/USDT",
    "ZKJ/USDT", "ENJ/USDT", "GALA/USDT", "APE/USDT", "NMR/USDT",
    "RON/USDT", "SAHARA/USDT", "ARG/USDT", "PYUSD/USDT",

    # Layer 2 / Infra
    "OP/USDT", "IMX/USDT", "ARB/USDT",
    "ZIL/USDT", "ZRX/USDT", "SKL/USDT", "DAI/USDT", "MAJOR/USDT", "CITY/USDT",

    # AI / Render / Web3
    "BAL/USDT", "CELR/USDT", "JUP/USDT", "SLP/USDT", "WOO/USDT",
    "KDA/USDT", "IOTA/USDT", "CHZ/USDT", "YGG/USDT", "BONK/USDT",
    "MERL/USDT",
    "GLMR/USDT", "T/USDT", "BADGER/USDT", "PHA/USDT", "KNC/USDT",
    "BICO/USDT", "LEO/USDT", "OKSOL/USDT", "NC/USDT", "VELO/USDT",
    "EGLD/USDT", "TRB/USDT",

    # Meme/Trendy
    "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "ASP/USDT", "WIF/USDT",
    "ORDI/USDT", "FLOKI/USDT", "NOT/USDT"
]

# ===============================
# ⏱ إعدادات التداول (تبقى كما هي)
# ===============================
TIMEFRAME = "5m"
TRADE_AMOUNT_USDT = 45
MAX_OPEN_POSITIONS = 3

# ===============================
# 🧮 الرسوم (round-trip) بالـ bps
# ===============================
FEE_BPS_ROUNDTRIP = 16

# ===============================
# ⚙️ خيارات التوسيع التلقائي
# ===============================
AUTO_EXPAND_SYMBOLS = bool(int(os.getenv("AUTO_EXPAND_SYMBOLS", "1")))  # عطّلها بوضع 0
TARGET_SYMBOLS_COUNT = int(os.getenv("TARGET_SYMBOLS_COUNT", "100"))
DEBUG_CONFIG_SYMBOLS = bool(int(os.getenv("DEBUG_CONFIG_SYMBOLS", "0")))

OKX_BASE = "https://www.okx.com"
TICKERS_URL = f"{OKX_BASE}/api/v5/market/tickers?instType=SPOT"  # يحوي vol24h/last... وهو كافٍ لرتبة السيولة
TIMEOUT_SEC = 12

def _normalize_symbol(s: str) -> str:
    return s.strip().upper().replace("-", "/")

def _dedupe_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _okx_get_json(url, attempts=3):
    for a in range(attempts):
        try:
            r = requests.get(url, timeout=TIMEOUT_SEC)
            if r.status_code == 429:
                # معدل زائد — backoff
                time.sleep((2 ** a) + random.random())
                continue
            r.raise_for_status()
            j = r.json()
            # بعض أخطاء OKX تأتي كـ code في الـ JSON
            if str(j.get("code", "0")) not in ("0", "200"):
                time.sleep((2 ** a) + random.random())
                continue
            return j
        except Exception:
            time.sleep((2 ** a) + random.random())
    return None

def _fetch_okx_usdt_spot_ranked():
    """يرجع قائمة أزواج SPOT/USDT من OKX مرتبة تقريبياً حسب السيولة (vol/last)."""
    j = _okx_get_json(TICKERS_URL, attempts=3)
    if not j:
        return []
    rows = []
    for it in j.get("data", []):
        inst = (it.get("instId") or "").upper()  # مثل BTC-USDT
        if not inst.endswith("-USDT"):
            continue
        # تقدير حجم/سيولة لعمل ترتيب تقريبي
        vol = 0.0
        for key in ("volCcy24h", "volUsd", "vol24h"):
            v = it.get(key)
            if v:
                try:
                    vol = float(v); break
                except:  # noqa
                    pass
        if vol == 0.0:
            try:
                vol = float(it.get("vol24h", 0)) * float(it.get("last", 0))
            except:  # noqa
                vol = 0.0
        sym = inst.replace("-", "/")
        rows.append((sym, vol))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows]

def _expand_symbols_to_target(existing, target=100):
    # طبّع وأزل التكرارات
    base = _dedupe_keep_order(_normalize_symbol(s) for s in existing)
    okx_ranked = _fetch_okx_usdt_spot_ranked()
    if not okx_ranked:
        # لو فشل الجلب: التزم بما لديك وقصّه للهدف
        out = base[:target]
        if DEBUG_CONFIG_SYMBOLS:
            print(f"[config] OKX fetch failed; using existing ({len(out)})")
        return out

    # احتفظ بما لديك لكن فقط إن كان مدعومًا على OKX
    okx_set = set(okx_ranked)
    kept = [s for s in base if s in okx_set]

    # أكمل من الأعلى سيولةً مع استبعاد الموجود
    extras = [s for s in okx_ranked if s not in kept]
    out = (kept + extras)[:target]

    if DEBUG_CONFIG_SYMBOLS:
        missing = [s for s in base if s not in okx_set]
        print(f"[config] kept {len(kept)}, added {len(out)-len(kept)}, missing_from_okx: {missing[:10]}")
    return out

if AUTO_EXPAND_SYMBOLS:
    try:
        SYMBOLS = _expand_symbols_to_target(SYMBOLS, TARGET_SYMBOLS_COUNT)
    except Exception as _e:
        # لو حدث خطأ غير متوقع: حافظ على قائمتك بعد إزالة التكرارات واقصّها
        SYMBOLS = _dedupe_keep_order(_normalize_symbol(s) for s in SYMBOLS)[:TARGET_SYMBOLS_COUNT]
else:
    # حتى بدون جلب، نظّف التكرارات وطبّع الرموز واقصّها
    SYMBOLS = _dedupe_keep_order(_normalize_symbol(s) for s in SYMBOLS)[:TARGET_SYMBOLS_COUNT]
