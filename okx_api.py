# -*- coding: utf-8 -*-
# okx_api.py — متوافق مع strategy.py (Spot فقط) + كاش أسعار جماعي + أدوات مساعدة
# المتطلبات: pip install ccxt requests

import os
import time
import random
import threading
from typing import Optional, Tuple, Dict, Any, List

import requests
import ccxt

# ================= مفاتيح OKX (اسم موحّد عبر المشروع) =================
# نحاول أولاً من config.py (بنفس الأسماء)، وإن لم تتوفر نقرأ من متغيرات البيئة
try:
    from config import OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE
except Exception:
    OKX_API_KEY = os.getenv("OKX_API_KEY", "")
    OKX_API_SECRET = os.getenv("OKX_API_SECRET", "")
    OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# ================= إعداد واجهة OKX ================
exchange = ccxt.okx({
    "apiKey": OKX_API_KEY,
    "secret": OKX_API_SECRET,
    "password": OKX_PASSPHRASE,   # مهم في OKX
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
        # بعض البورصات تتطلب price مع أوامر Market Buy — OKX لا يتطلب عادةً، نخليه False
        "createMarketBuyOrderRequiresPrice": False,
    },
    "timeout": 15000,  # 15s
})

# تفعيل الساندبوكس عند الحاجة
if os.getenv("OKX_TESTNET", "false").lower() in ("1", "true", "yes"):
    try:
        exchange.set_sandbox_mode(True)
        print("ℹ️ OKX Sandbox mode: ON")
    except Exception:
        pass

# تحميل الأسواق مرة واحدة
try:
    exchange.load_markets()
except Exception as e:
    print(f"⚠️ تعذّر تحميل الأسواق: {e}")

OKX_BASE = "https://www.okx.com"
OKX_TICKERS_URL = f"{OKX_BASE}/api/v5/market/tickers?instType=SPOT"

# ================ أدوات مساعدة عامة ================
def _fmt_symbol(symbol: str) -> str:
    """
    يطبع رموز مثل BTC-USDT, BTC_USDT, BTC/USDT#new → BTC/USDT
    (يزيل لاحقة الاستراتيجية مثل #old/#new إن وُجدت).
    """
    s = (symbol or "").strip()
    if "#" in s:
        s = s.split("#", 1)[0]
    return s.replace("-", "/").replace("_", "/").upper()

def _okx_error_hint(e: Exception) -> str:
    msg = str(e)
    if "50110" in msg or ("IP" in msg and "whitelist" in msg.lower()):
        return "❗️يبدو أن IP غير مُدرج في قائمة السماح لـ OKX (50110). أضف IP خادمك في إعدادات مفاتيح OKX."
    if "Insufficient" in msg or "insufficient" in msg:
        return "⚠️ رصيد غير كافٍ لإتمام العملية."
    if "Rate limit" in msg or "Too Many Requests" in msg:
        return "⏳ تم تجاوز حد المعدّل، أعد المحاولة بعد لحظات."
    return ""

def _log(send_message, text: str):
    print(text)
    if send_message:
        try:
            send_message(text)
        except Exception:
            pass

# ================ Backoff + Jitter ================
def _retry(times=3, base_delay=0.6, max_delay=5.0):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last_exc = None
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    hint = _okx_error_hint(e)
                    if hint:
                        print(hint)
                    sleep_s = min(max_delay, base_delay * (1.8 ** i)) + random.uniform(0, 0.25)
                    time.sleep(sleep_s)
            raise last_exc
        return wrapper
    return deco

# ================ كاش بسيط ================
_BAL_CACHE: Dict[str, Tuple[float, float]] = {}                 # {asset: (ts, balance)}
_TICKER_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}     # {sym: (ts, ticker-like)}
CACHE_TTL_SEC_BAL = 3.0
CACHE_TTL_SEC_TICKER = 2.0

def _get_cached_bal(asset: str) -> Optional[float]:
    t = _BAL_CACHE.get(asset)
    if not t:
        return None
    ts, val = t
    return val if (time.time() - ts) <= CACHE_TTL_SEC_BAL else None

def _set_cached_bal(asset: str, val: float):
    _BAL_CACHE[asset] = (time.time(), val)

def _get_cached_ticker(sym: str) -> Optional[Dict[str, Any]]:
    t = _TICKER_CACHE.get(sym)
    if not t:
        return None
    ts, data = t
    return data if (time.time() - ts) <= CACHE_TTL_SEC_TICKER else None

def _set_cached_ticker(sym: str, data: Dict[str, Any]):
    _TICKER_CACHE[sym] = (time.time(), data)

# ================ كاش أسعار جماعي (اختياري) ================
_cache_thread: Optional[threading.Thread] = None
_cache_stop = False

def _refresh_tickers_loop(period=3, usdt_only=True):
    global _cache_stop
    while not _cache_stop:
        try:
            r = requests.get(OKX_TICKERS_URL, timeout=10)
            if r.status_code == 429:
                time.sleep(period + random.random()); continue
            j = r.json()
            for it in j.get("data", []):
                inst = str(it.get("instId", "")).upper()  # BTC-USDT
                if usdt_only and not inst.endswith("-USDT"):
                    continue
                sym = inst.replace("-", "/")
                last = it.get("last") or it.get("close") or it.get("ask") or it.get("bid")
                try:
                    last_f = float(last or 0.0)
                except Exception:
                    last_f = 0.0
                # نبني شكل يشبه fetch_ticker حتى تتوافق الدوال
                _set_cached_ticker(sym, {"symbol": sym, "last": last_f})
        except Exception:
            pass
        time.sleep(max(1, int(period)))

def start_tickers_cache(period: int = 3, usdt_only: bool = True):
    """ابدأ تحديث الأسعار جماعياً كل period ثوانٍ (طلب واحد لكل الدورة)."""
    global _cache_thread, _cache_stop
    if _cache_thread and _cache_thread.is_alive():
        return
    _cache_stop = False
    _cache_thread = threading.Thread(target=_refresh_tickers_loop, args=(period, usdt_only), daemon=True)
    _cache_thread.start()
    print(f"✓ OKX tickers cache started (period={period}s, usdt_only={usdt_only})")

def stop_tickers_cache():
    """أوقف مؤقّت الأسعار الجماعي."""
    global _cache_stop
    _cache_stop = True

# ================ تسوية الكمية/الحدود ================
def _decimals_to_step(decimals: Optional[int]) -> Optional[float]:
    try:
        if decimals is None:
            return None
        return float(10 ** (-int(decimals)))
    except Exception:
        return None

def _amount_to_precision(symbol_ccxt: str, amount: float) -> float:
    """
    يضبط الكمية حسب دقة السوق وحدوده (min amount/min cost).
    """
    try:
        market = exchange.market(symbol_ccxt)
    except Exception:
        market = None

    try:
        amt = float(exchange.amount_to_precision(symbol_ccxt, amount))
    except Exception:
        amt = float(amount)

    # حد أدنى للكمية
    try:
        min_amt = float(market.get("limits", {}).get("amount", {}).get("min") or 0.0) if market else 0.0
        if min_amt and amt < min_amt:
            amt = float(exchange.amount_to_precision(symbol_ccxt, min_amt))
    except Exception:
        pass

    # حد أدنى للتكلفة
    try:
        min_cost = float(market.get("limits", {}).get("cost", {}).get("min") or 0.0) if market else 0.0
        if min_cost > 0:
            tkr = _get_cached_ticker(symbol_ccxt)
            if not tkr:
                try:
                    tkr = exchange.fetch_ticker(symbol_ccxt)
                    _set_cached_ticker(symbol_ccxt, tkr)
                except Exception:
                    tkr = None
            last = 0.0
            if tkr:
                last = float(tkr.get("last") or tkr.get("close") or tkr.get("ask") or tkr.get("bid") or 0.0)
            if last > 0 and amt * last < min_cost:
                needed = min_cost / last
                try:
                    amt = float(exchange.amount_to_precision(symbol_ccxt, needed))
                except Exception:
                    amt = float(needed)
    except Exception:
        pass

    return max(0.0, amt)

# ================ واجهات مطلوبة من strategy.py ================
@_retry()
def fetch_balance(asset: str = "USDT") -> float:
    """يعيد الرصيد (Free) للعملة المطلوبة. يستخدم كاش قصير لتخفيف الضغط."""
    asset = (asset or "USDT").upper()
    cached = _get_cached_bal(asset)
    if cached is not None:
        return cached
    try:
        balances = exchange.fetch_balance()
        free = balances.get("free", {}) or {}
        val = float(free.get(asset, 0.0) or 0.0)
        _set_cached_bal(asset, val)
        return val
    except Exception as e:
        print(f"❌ خطأ في جلب الرصيد لـ {asset}: {e}")
        return 0.0

@_retry()
def fetch_price(symbol: str) -> float:
    """يعيد آخر سعر (يحاول من الكاش الجماعي ثم CCXT)."""
    sym = _fmt_symbol(symbol)
    cached = _get_cached_ticker(sym)
    if cached:
        price = cached.get("last") or cached.get("close") or cached.get("ask") or cached.get("bid")
        try:
            return float(price or 0.0)
        except Exception:
            pass
    try:
        t = exchange.fetch_ticker(sym)
        _set_cached_ticker(sym, t)
        price = t.get("last") or t.get("close") or t.get("ask") or t.get("bid")
        return float(price or 0.0)
    except Exception as e:
        print(f"❌ خطأ في جلب السعر الحالي لـ {sym}: {e}")
        return 0.0

@_retry()
def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 100):
    """
    يعيد بيانات الشموع بصيغة CCXT: [[ts, open, high, low, close, volume], ...]
    NOTE: OKX يدعم حدودًا مختلفة حسب الإطار؛ 200–300 آمن عادةً.
    """
    sym = _fmt_symbol(symbol)
    limit = max(10, min(int(limit or 100), 500))
    try:
        data = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        return data or []
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات الشموع لـ {sym}({timeframe}, {limit}): {e}")
        return []

@_retry()
def place_market_order(symbol: str, side: str, amount: float, send_message=None):
    """
    ينفّذ أمر سوق (شراء/بيع) مع تسوية الكمية وفق حدود السوق.
    يعيد أمر CCXT عند النجاح أو None عند الفشل.
    """
    sym = _fmt_symbol(symbol)
    side = (side or "").lower().strip()
    if side not in ("buy", "sell"):
        _log(send_message, f"⚠️ side غير صحيح ({side}) — يجب buy أو sell")
        return None

    if amount is None or amount <= 0:
        _log(send_message, f"⚠️ كمية غير صالحة لأمر {side} على {symbol}")
        return None

    adj_amount = _amount_to_precision(sym, float(amount))
    if adj_amount <= 0:
        _log(send_message, f"⚠️ الكمية بعد التسوية أصبحت صفر لـ {symbol} — تحقق من الحد الأدنى للسوق.")
        return None

    try:
        # params إضافية اختيارية لـ OKX: {"tgtCcy": "base_ccy"} عند الحاجة
        order = exchange.create_order(sym, type="market", side=side, amount=adj_amount)
        _log(send_message, f"✅ تم تنفيذ أمر {side.UPPER()} لـ {symbol} (كمية: {adj_amount})")
        return order
    except Exception as e:
        hint = _okx_error_hint(e)
        if hint:
            _log(send_message, hint)
        _log(send_message, f"❌ فشل تنفيذ أمر السوق ({side}) لـ {symbol}: {str(e)}")
        return None

# ================ أدوات إضافية مفيدة ================
@_retry()
def list_okx_usdt_spot_symbols() -> List[str]:
    """
    يرجع قائمة كل أزواج SPOT/USDT المدعومة على OKX بصيغة BTC/USDT.
    مفيد لفلترة/توسيع SYMBOLS خارج هذا الملف.
    """
    try:
        r = requests.get(OKX_TICKERS_URL, timeout=12)
        r.raise_for_status()
        j = r.json()
        out = []
        for it in j.get("data", []):
            inst = str(it.get("instId", "")).upper()  # BTC-USDT
            if inst.endswith("-USDT"):
                out.append(inst.replace("-", "/"))
        # إزالة التكرار مع الحفاظ على الترتيب
        seen, uniq = set(), []
        for s in out:
            if s not in seen:
                uniq.append(s); seen.add(s)
        return uniq
    except Exception as e:
        print(f"⚠️ فشل جلب قائمة USDT/Spot: {e}")
        return []

# ================== fetch_symbol_filters (مصَحَّحة) ==================
def fetch_symbol_filters(symbol: str) -> Dict[str, float]:
    """
    إرجاع فلاتر الرمز (minQty, minNotional, stepSize, tickSize)
    لضمان تنفيذ صحيح في أوامر الشراء/البيع.
    نعتمد أولاً على ccxt.market(...) ثم نوفّر بدائل آمنة.
    """
    sym = _fmt_symbol(symbol)
    try:
        m = exchange.market(sym)  # يقرأ من الأسواق المحمّلة
    except Exception as e:
        print(f"[fetch_symbol_filters] market() error {sym}: {e}")
        m = None

    # قيم افتراضية آمنة
    step_size = 0.000001
    tick_size = 0.000001
    min_qty = 0.0
    min_notional = 0.0

    if m:
        # stepSize: من lotSz إن توفرت، وإلا من precision.amount (عدد الخانات)
        try:
            lot_sz = float(m.get("info", {}).get("lotSz", 0) or 0)
            if lot_sz and lot_sz > 0:
                step_size = lot_sz
            else:
                dec = m.get("precision", {}).get("amount")
                step = _decimals_to_step(dec)
                if step:
                    step_size = step
        except Exception:
            pass

        # tickSize: من tickSz أو من precision.price
        try:
            tick_sz = float(m.get("info", {}).get("tickSz", 0) or 0)
            if tick_sz and tick_sz > 0:
                tick_size = tick_sz
            else:
                pdec = m.get("precision", {}).get("price")
                step = _decimals_to_step(pdec)
                if step:
                    tick_size = step
        except Exception:
            pass

        # minQty: من limits.amount.min أو minSz
        try:
            min_qty = float(
                m.get("limits", {}).get("amount", {}).get("min")
                or m.get("info", {}).get("minSz")
                or 0.0
            )
        except Exception:
            min_qty = 0.0

        # minNotional (min cost): من limits.cost.min؛ وإلا تقدير = minQty * last
        try:
            min_cost = m.get("limits", {}).get("cost", {}).get("min")
            if min_cost is not None:
                min_notional = float(min_cost or 0.0)
            else:
                last = fetch_price(sym)
                min_notional = float(min_qty * last) if last > 0 else 0.0
        except Exception:
            pass

    return {
        "minQty": float(max(0.0, min_qty)),
        "minNotional": float(max(0.0, min_notional)),
        "stepSize": float(max(1e-12, step_size)),
        "tickSize": float(max(1e-12, tick_size)),
    }
