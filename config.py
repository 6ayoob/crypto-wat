
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
# 🔑 مفاتيح API لـ OKX (استخدم متغيرات بيئة للأمان)
# ===============================
# ===============================
# 📈 الرموز — قائمة منقّحة (سيتم فلترتها وإكمالها تلقائياً من OKX عند الإقلاع)
# ملاحظة: لا تعتمد على هذه القائمة فقط؛ سيتم التحقق من دعم OKX وإكمالها حتى 100.
# ===============================

import os, time, random, re
import requests
from typing import List, Tuple, Optional

# قائمتك الأصلية (ستُطبّع للأحرف الكبيرة وتزال التكرارات)
SEED_SYMBOLS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT","TRX/USDT","TON/USDT","DOT/USDT",
    "AVAX/USDT","MAJOR/USDT","LINK/USDT","LTC/USDT","BCH/USDT","ETC/USDT","NEAR/USDT","ATOM/USDT","TIA/USDT","ARB/USDT",
    "OP/USDT","STRK/USDT","ZK/USDT","SUI/USDT","APT/USDT","DUCK/USDT","INJ/USDT","STX/USDT","GOAT/USDT","HUMA/USDT",
    "PYTH/USDT","JTO/USDT","JUP/USDT","W/USDT","TNSR/USDT","WIF/USDT","BONK/USDT","PEPE/USDT","FLOKI/USDT","SHIB/USDT",
    "MEME/USDT","KAIA/USDT","AAVE/USDT","UNI/USDT","SUSHI/USDT","COMP/USDT","BABY/USDT","SNX/USDT","LDO/USDT","CRV/USDT",
    "BAL/USDT","YFI/USDT","CAT/USDT","GMX/USDT","DYDX/USDT","1INCH/USDT","CVX/USDT","RPL/USDT","FXS/USDT","SSV/USDT",
    "LQTY/USDT","APE/USDT","ENS/USDT","GRT/USDT","CHZ/USDT","AXS/USDT","SAND/USDT","MANA/USDT","IMX/USDT","GALA/USDT",
    "RON/USDT","WLFI/USDT","MAGIC/USDT","PARTI/USDT","SD/USDT","MINA/USDT","ICP/USDT","FIL/USDT","AR/USDT","STORJ/USDT",
    "SC/USDT","OKB/USDT","HBAR/USDT","EGLD/USDT","ALGO/USDT","CTC/USDT","THETA/USDT","CFX/USDT","XTZ/USDT","ZIL/USDT",
    "NEO/USDT","QTUM/USDT","IOTA/USDT","ONDO/USDT","FXS/USDT","POL/USDT","RPL/USDT","ETHFI/USDT","PENDLE/USDT","ZIL/USDT",
    "AEVO/USDT","ZRO/USDT","BOME/USDT","ORDI/USDT","SATS/USDT","JST/USDT","CFX/USDT","CELO/USDT","CKB/USDT","ROSE/USDT",
    "AKT/USDT","TWT/USDT","ZEC/USDT","SXP/USDT","HFT/USDT","C98/USDT","PEOPLE/USDT","LUNC/USDT","LUNA/USDT","STG/USDT",
    "ELF/USDT","SKL/USDT","COTI/USDT","OGN/USDT","ID/USDT","EOS/USDT","ETHW/USDT","FLOW/USDT","FTM/USDT","GAL/USDT",
    "GLMR/USDT","GMT/USDT","IOST/USDT","JASMY/USDT","JOE/USDT","KLAY/USDT","KNC/USDT","KSM/USDT","LINA/USDT","LPT/USDT",
    "LRC/USDT","MASK/USDT","MAV/USDT","NOT/USDT","OKB/USDT","OM/USDT","ONT/USDT","POLS/USDT","PRIME/USDT","RSR/USDT",
    "SFP/USDT","SLERF/USDT","TRB/USDT","UMA/USDT","UNFI/USDT","USDC/USDT","WLD/USDT","WOO/USDT","XAI/USDT","XLM/USDT",
    "YGG/USDT","ZETA/USDT","ZRX/USDT","ZBCN/USDT","BLUR/USDT","BNT/USDT","BICO/USDT","BIGTIME/USDT","CELR/USDT","CETUS/USDT",
    "CFG/USDT","CYBER/USDT","DODO/USDT","DYM/USDT","EDU/USDT","EUL/USDT","FIDA/USDT","FLM/USDT","FRONT/USDT","FX/USDT","LAYER/USDT","LQTY/USDT",
    "GHST/USDT","HIGH/USDT","HOOK/USDT","HOT/USDT","IDEX/USDT","ILV/USDT","LOKA/USDT","MBOX/USDT","NKN/USDT","NMR/USDT","CFG/USDT","OL/USDT",
    "PERP/USDT","PHA/USDT","PLA/USDT","RLC/USDT","STMX/USDT","STPT/USDT","SUPER/USDT","SWEAT/USDT","SYS/USDT","TLM/USDT"
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

# === خيارات فلترة/سيولة قابلة للتهيئة (اختياري) ===
ALLOWED_QUOTE = os.getenv("ALLOWED_QUOTE", "USDT").upper()         # الاقتباس المسموح (افتراضي USDT)
MIN_USDT_VOL_24H = float(os.getenv("MIN_USDT_VOL_24H", "1000000")) # حد السيولة الأدنى بالدولار (افتراضي 1M)

# ✅ استبعاد يعتمد على BASE فقط (بدون ضرب أزواج /USDT)
EXCLUDE_BASE_REGEX = os.getenv("EXCLUDE_BASE_REGEX", r"(TEST|IOU)")
INCLUDE_REGEX = os.getenv("INCLUDE_REGEX", "")  # لتضمين رموز قسراً إن لزم

# قواعد استبعاد إضافية للـ BASE
_STABLE_BASES = {"USDT","USDC","DAI","FDUSD","TUSD","PYUSD","EUR","TRY","BRL","AED","GBP","JPY"}
_LEVERAGED_SUFFIXES = ("3L","3S","5L","5S")

OKX_TICKERS_CACHE_SEC = int(os.getenv("OKX_TICKERS_CACHE_SEC", "90"))

OKX_BASE = "https://www.okx.com"
TICKERS_URL = f"{OKX_BASE}/api/v5/market/tickers?instType=SPOT"     # يحوي vol24h/last... وهو كافٍ لرتبة السيولة
INSTR_URL  = f"{OKX_BASE}/api/v5/public/instruments?instType=SPOT"  # مسار احتياطي للتأكد من الدعم
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

# جلسة Requests واحدة مع UA وتوقيت
_REQ_SESSION: Optional[requests.Session] = None
def _get_session() -> requests.Session:
    global _REQ_SESSION
    if _REQ_SESSION is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "mk1-ai-bot/scan (okx liquidity filter)",
            "Accept": "application/json",
        })
        _REQ_SESSION = s
    return _REQ_SESSION

# كاش بسيط للتيكرز
_TICKERS_CACHE: Tuple[float, Optional[dict]] = (0.0, None)

def _okx_get_json(url, attempts=3):
    sess = _get_session()
    for a in range(attempts):
        try:
            r = sess.get(url, timeout=TIMEOUT_SEC)
            if r.status_code == 429:
                time.sleep((2 ** a) + random.random())  # backoff مع jitter
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

def _okx_get_tickers_cached():
    global _TICKERS_CACHE
    ts, cached = _TICKERS_CACHE
    now = time.time()
    if cached and (now - ts) < OKX_TICKERS_CACHE_SEC:
        return cached
    j = _okx_get_json(TICKERS_URL, attempts=3)
    if j:
        _TICKERS_CACHE = (now, j)
    return j

_BASE_EXCLUDE_RE = re.compile(EXCLUDE_BASE_REGEX, re.IGNORECASE) if EXCLUDE_BASE_REGEX else None
_INCLUDE_FORCE = re.compile(INCLUDE_REGEX, re.IGNORECASE) if INCLUDE_REGEX else None

def _looks_stable_or_exotic(symbol: str) -> bool:
    """فلترة تستهدف BASE فقط حتى لا نستبعد أزواج /USDT بالخطأ."""
    # تضمين قسري إن طابق INCLUDE_REGEX
    if _INCLUDE_FORCE and _INCLUDE_FORCE.search(symbol):
        return False
    try:
        base, quote = symbol.split("/", 1)
    except ValueError:
        base, quote = symbol, ""
    base = base.upper().strip()
    quote = quote.upper().strip()

    # استبعاد إذا كان الـ BASE نفسه ستايبل/فيات
    if base in _STABLE_BASES:
        return True

    # استبعاد توكنات مرفوعة مثل BTC3L/USDT
    if any(base.endswith(suf) for suf in _LEVERAGED_SUFFIXES):
        return True

    # استبعاد أسماء BASE الغريبة (TEST|IOU...) إن طابقت
    if _BASE_EXCLUDE_RE and _BASE_EXCLUDE_RE.search(base):
        return True

    return False

def _fetch_okx_spot_supported() -> List[str]:
    """قائمة جميع أزواج SPOT المدعومة (بدون ترتيب)."""
    j = _okx_get_json(INSTR_URL, attempts=2)
    if not j:
        return []
    out = []
    for it in j.get("data", []):
        inst = (it.get("instId") or "").upper()  # BTC-USDT
        if not inst or f"-{ALLOWED_QUOTE}" not in inst:
            continue
        sym = inst.replace("-", "/")
        if not _looks_stable_or_exotic(sym):
            out.append(sym)
    return out

def _fetch_okx_usdt_spot_ranked(min_usd_vol: float) -> List[Tuple[str, float]]:
    """
    يرجع [(symbol, score)] مرتبة تقريبياً حسب السيولة (vol/last).
    يتم تجاهل الأزواج المستبعدة، ولا تُرجع رموز تحت حد السيولة.
    """
    j = _okx_get_tickers_cached()
    if not j:
        return []
    rows: List[Tuple[str, float]] = []
    for it in j.get("data", []):
        inst = (it.get("instId") or "").upper()  # مثل BTC-USDT
        if not inst.endswith(f"-{ALLOWED_QUOTE}"):
            continue

        # حساب تقدير الحجم/السيولة بالدولار
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
        if vol <= 0:
            try:
                vol = float(it.get("vol24h", 0)) * float(it.get("last", 0))
            except Exception:
                vol = 0.0

        sym = inst.replace("-", "/")
        if _looks_stable_or_exotic(sym):
            continue
        if vol < min_usd_vol:
            continue
        rows.append((sym, vol))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows

def _expand_symbols_to_target(existing: List[str], target=100) -> List[str]:
    # طبّع وأزل التكرارات
    base = _dedupe_keep_order(_normalize_symbol(s) for s in existing)

    # 1) ترتيب بالسيولة
    ranked = _fetch_okx_usdt_spot_ranked(MIN_USDT_VOL_24H)
    if ranked:
        okx_ranked = [s for s, _ in ranked]
        okx_set = set(okx_ranked)

        # احتفظ بما لديك إن كان مدعومًا ومقبولًا
        kept = [s for s in base if (s in okx_set) and not _looks_stable_or_exotic(s)]

        # أكمل من الأعلى سيولةً مع استبعاد الموجود
        extras = [s for s in okx_ranked if s not in kept]
        out = (kept + extras)[:target]

        if DEBUG_CONFIG_SYMBOLS:
            missing = [s for s in base if s not in okx_set or _looks_stable_or_exotic(s)]
            print(f"[config] kept {len(kept)}, added {len(out)-len(kept)}, missing_or_filtered: {missing[:10]}")
        return out

    # 2) مسار احتياطي: instruments (بدون ترتيب)
    supported = set(_fetch_okx_spot_supported())
    if supported:
        kept = [s for s in base if s in supported and not _looks_stable_or_exotic(s)]
        extras = [s for s in supported if (s not in kept) and not _looks_stable_or_exotic(s)]
        out = (kept + extras)[:target]
        if DEBUG_CONFIG_SYMBOLS:
            print(f"[config] tickers failed; used instruments. kept={len(kept)} total={len(out)}")
        return out

    # 3) فشل كامل: التزم بما لديك بعد الفلترة
    out = [s for s in base if not _looks_stable_or_exotic(s)][:target]
    if DEBUG_CONFIG_SYMBOLS:
        print(f"[config] OKX fetch failed; using existing ({len(out)})")
    return out

# ============ التنفيذ ============

try:
    if AUTO_EXPAND_SYMBOLS:
        SYMBOLS = _expand_symbols_to_target(SEED_SYMBOLS, TARGET_SYMBOLS_COUNT)
    else:
        # حتى بدون جلب، نظّف التكرارات وطبّع الرموز واقصّها واستبعد الـ BASE غير المرغوب
        SYMBOLS = [
            s for s in _dedupe_keep_order(_normalize_symbol(s) for s in SEED_SYMBOLS)
            if not _looks_stable_or_exotic(s)
        ][:TARGET_SYMBOLS_COUNT]
except Exception:
    # لو حدث خطأ غير متوقع: حافظ على قائمتك بعد إزالة التكرارات واقصّها
    SYMBOLS = _dedupe_keep_order(_normalize_symbol(s) for s in SEED_SYMBOLS)
    SYMBOLS = list(SYMBOLS)[:TARGET_SYMBOLS_COUNT]

if DEBUG_CONFIG_SYMBOLS:
    print(f"[config] SYMBOLS ready: {len(SYMBOLS)} | first 10: {SYMBOLS[:10]}")

# ===============================
# 🎯 نسختين لأول 20 عملة والباقي نسخة واحدة
# ===============================
FIRST_N_FOR_BOTH = 20
_final_symbols = []
for idx, s in enumerate(SYMBOLS):
    _final_symbols.append(s)                 # النسخة الافتراضية (#new)
    if idx < FIRST_N_FOR_BOTH:
        _final_symbols.append(f"{s}#old")    # أضف نسخة #old لأول 20

SYMBOLS = _final_symbols

if DEBUG_CONFIG_SYMBOLS:
    print(f"[config] final SYMBOLS: {len(SYMBOLS)} | first 12: {SYMBOLS[:12]}")
