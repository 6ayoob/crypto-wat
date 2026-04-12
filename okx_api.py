# -*- coding: utf-8 -*-
"""
okx_api.py — نسخة محسّنة v2.0
التغييرات الرئيسية:
- إصلاح average بعد الشراء: انتظار تأكيد fill أو fallback ذكي
- توحيد Retry: إزالة @_retry من fetch_ohlcv (strategy.py عنده retry خاص)
- تقليل CACHE_TTL للسعر: 6s → 3s لـ SL أدق
- إضافة timestamp validation لـ OHLCV
- fetch_symbol_filters موثوقة من exchange.market()
"""

import os
import time
import random
import threading
from typing import Optional, Tuple, Dict, Any, List

import requests
import ccxt

# ================= مفاتيح OKX =================
try:
    from config import OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE
except Exception:
    OKX_API_KEY    = os.getenv("OKX_API_KEY",    "")
    OKX_API_SECRET = os.getenv("OKX_API_SECRET", "")
    OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# ================= إعداد OKX =================
exchange = ccxt.okx({
    "apiKey":   OKX_API_KEY,
    "secret":   OKX_API_SECRET,
    "password": OKX_PASSPHRASE,
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
        "createMarketBuyOrderRequiresPrice": False,
    },
    "timeout": 15000,
})

if os.getenv("OKX_TESTNET", "false").lower() in ("1", "true", "yes"):
    try:
        exchange.set_sandbox_mode(True)
        print("ℹ️ OKX Sandbox: ON", flush=True)
    except Exception:
        pass

try:
    exchange.load_markets()
except Exception as e:
    print(f"⚠️ تعذّر تحميل الأسواق: {e}", flush=True)

OKX_BASE      = "https://www.okx.com"
TICKERS_URL   = f"{OKX_BASE}/api/v5/market/tickers?instType=SPOT"

# ================ كاش =================
_BAL_CACHE:    Dict[str, Tuple[float, float]] = {}
_TICKER_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# ← تقليل TTL للسعر من 6s إلى 3s لـ SL أدق
CACHE_TTL_BAL    = float(os.getenv("CACHE_TTL_BAL",    "3.0"))
CACHE_TTL_TICKER = float(os.getenv("CACHE_TTL_TICKER", "3.0"))

def _get_cached_bal(asset: str) -> Optional[float]:
    t = _BAL_CACHE.get(asset)
    if not t:
        return None
    ts, val = t
    return val if (time.time() - ts) <= CACHE_TTL_BAL else None

def _set_cached_bal(asset: str, val: float) -> None:
    _BAL_CACHE[asset] = (time.time(), val)

def _get_cached_ticker(sym: str) -> Optional[Dict[str, Any]]:
    t = _TICKER_CACHE.get(sym)
    if not t:
        return None
    ts, data = t
    return data if (time.time() - ts) <= CACHE_TTL_TICKER else None

def _set_cached_ticker(sym: str, data: Dict[str, Any]) -> None:
    _TICKER_CACHE[sym] = (time.time(), data)

# ================ كاش أسعار جماعي =================
_cache_thread: Optional[threading.Thread] = None
_cache_stop = False

def _refresh_tickers_loop(period: int = 5, usdt_only: bool = True) -> None:
    global _cache_stop
    while not _cache_stop:
        try:
            r = requests.get(TICKERS_URL, timeout=10)
            if r.status_code == 429:
                time.sleep(period + random.random())
                continue
            j = r.json()
            for it in j.get("data", []):
                inst = str(it.get("instId", "")).upper()
                if usdt_only and not inst.endswith("-USDT"):
                    continue
                sym = inst.replace("-", "/")
                last = it.get("last") or it.get("close") or it.get("ask") or it.get("bid")
                try:
                    _set_cached_ticker(sym, {"symbol": sym, "last": float(last or 0.0)})
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(max(1, int(period)))

def start_tickers_cache(period: int = 5, usdt_only: bool = True) -> None:
    global _cache_thread, _cache_stop
    if _cache_thread and _cache_thread.is_alive():
        return
    _cache_stop = False
    _cache_thread = threading.Thread(
        target=_refresh_tickers_loop,
        args=(period, usdt_only),
        daemon=True,
    )
    _cache_thread.start()
    print(f"✓ OKX tickers cache started (period={period}s)", flush=True)

def stop_tickers_cache() -> None:
    global _cache_stop
    _cache_stop = True

# ================ أدوات مساعدة =================
def _fmt_symbol(symbol: str) -> str:
    s = (symbol or "").strip()
    if "#" in s:
        s = s.split("#", 1)[0]
    return s.replace("-", "/").replace("_", "/").upper()

def _okx_error_hint(e: Exception) -> str:
    msg = str(e)
    if "50110" in msg or ("IP" in msg and "whitelist" in msg.lower()):
        return "❗ IP غير مُدرج في قائمة السماح لـ OKX (50110)."
    if "Insufficient" in msg or "insufficient" in msg:
        return "⚠️ رصيد غير كافٍ."
    if "Rate limit" in msg or "Too Many Requests" in msg or "50011" in msg:
        return "⏳ Rate limit — أعد المحاولة بعد لحظات."
    return ""

def _decimals_to_step(decimals) -> Optional[float]:
    try:
        return float(10 ** (-int(decimals))) if decimals is not None else None
    except Exception:
        return None

# ================ Retry decorator (خفيف) =================
def _retry(times: int = 3, base_delay: float = 0.8, max_delay: float = 8.0):
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
                        print(hint, flush=True)
                    # تأخير أطول عند Rate Limit
                    boost = 3.0 if ("50011" in str(e) or "Too Many" in str(e)) else 1.0
                    wait  = min(max_delay, base_delay * (2 ** i) * boost)
                    wait += random.uniform(0, 0.3)
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return deco

# ================ واجهات strategy.py =================

@_retry(times=3)
def fetch_balance(asset: str = "USDT") -> float:
    asset  = (asset or "USDT").upper()
    cached = _get_cached_bal(asset)
    if cached is not None:
        return cached
    try:
        balances = exchange.fetch_balance()
        free = balances.get("free", {}) or {}
        val  = float(free.get(asset, 0.0) or 0.0)
        _set_cached_bal(asset, val)
        return val
    except Exception as e:
        print(f"❌ fetch_balance {asset}: {e}", flush=True)
        return 0.0

def fetch_price(symbol: str) -> float:
    """
    سعر آمن وسريع:
    1) كاش جماعي (< 3s)
    2) طلب منفرد fallback
    لا Retry هنا لتجنب تراكم الطلبات عند Rate Limit.
    """
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
        msg = str(e)
        if "50011" in msg or "Too Many" in msg:
            print(f"⚠️ fetch_price rate-limit {sym}", flush=True)
        else:
            print(f"❌ fetch_price {sym}: {e}", flush=True)
        return 0.0

# ← بدون @_retry هنا: strategy.py عنده _retry_fetch_ohlcv خاص به (5 محاولات)
# وضع @_retry هنا أيضاً كان يعني 15 محاولة = Rate Limit مضمون
def fetch_ohlcv(symbol: str, timeframe: str = "15m", limit: int = 120) -> list:
    """
    شموع OHLCV مع timestamp validation.
    بدون Retry — strategy.py يتولى إعادة المحاولة.
    """
    sym   = _fmt_symbol(symbol)
    limit = max(10, min(int(limit or 120), 500))
    try:
        data = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        if not data:
            return []

        # ← timestamp validation: تأكد أن البيانات حديثة
        _tf_minutes = {
            "1m":1,"3m":3,"5m":5,"15m":15,"30m":30,
            "1h":60,"2h":120,"4h":240,"1d":1440
        }
        tf_min  = _tf_minutes.get(timeframe.lower(), 60)
        max_age = tf_min * 3 * 60 * 1000  # 3 شموع بالـ ms
        now_ms  = int(time.time() * 1000)

        last_ts = data[-1][0]
        if last_ts < 10**12:
            last_ts *= 1000

        if (now_ms - last_ts) > max_age:
            print(f"⚠️ fetch_ohlcv {sym} {timeframe}: بيانات قديمة ({(now_ms-last_ts)//60000} دقيقة)", flush=True)
            # نرجع البيانات مع تحذير بدل رجوع قائمة فارغة
            # (strategy لديها فلتر _row_is_recent_enough)

        return data
    except Exception as e:
        print(f"❌ fetch_ohlcv {sym}({timeframe},{limit}): {e}", flush=True)
        return []

def place_market_order(symbol: str, side: str, amount: float, send_message=None):
    """
    تنفيذ أمر سوق مع:
    - تصحيح الكمية
    - انتظار تأكيد fill (average) حتى 3 ثوانٍ
    - fallback للسعر الحالي إن لم يُتاح average
    """
    sym  = _fmt_symbol(symbol)
    side = (side or "").lower().strip()

    if side not in ("buy", "sell"):
        print(f"⚠️ side غير صحيح ({side})", flush=True)
        return None

    if not amount or amount <= 0:
        print(f"⚠️ كمية غير صالحة لـ {symbol}", flush=True)
        return None

    # تصحيح الكمية
    try:
        market = exchange.market(sym)
    except Exception:
        market = None

    adj_amount = _adjust_amount(sym, float(amount), market, enforce_min_cost=(side == "buy"))
    if adj_amount <= 0:
        print(f"⚠️ الكمية بعد التسوية = 0 لـ {symbol}", flush=True)
        return None

    try:
        order = exchange.create_order(sym, type="market", side=side, amount=adj_amount)
    except Exception as e:
        hint = _okx_error_hint(e)
        if hint:
            print(hint, flush=True)
        print(f"❌ فشل أمر {side} {symbol}: {e}", flush=True)
        return None

    if not order:
        return None

    # ← إصلاح average: OKX أحياناً يُرجع average=None فوراً
    # ننتظر حتى 3 ثوانٍ للحصول على average حقيقي
    order = _ensure_fill_price(sym, order, timeout_sec=3.0)

    print(f"✅ {side.upper()} {symbol} qty={adj_amount} avg={order.get('average','?')}", flush=True)
    return order

def _ensure_fill_price(sym: str, order: dict, timeout_sec: float = 3.0) -> dict:
    """
    إذا average فارغ → نحاول جلب تفاصيل الأمر حتى timeout_sec.
    Fallback: نستخدم fetch_price الحالي.
    """
    if order.get("average") and float(order.get("average") or 0) > 0:
        return order  # average موجود، لا حاجة لانتظار

    order_id = order.get("id")
    if not order_id:
        # لا id → نستخدم السعر الحالي كـ fallback
        px = fetch_price(sym)
        if px > 0:
            order["average"] = px
        return order

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            details = exchange.fetch_order(order_id, sym)
            avg = details.get("average") or details.get("price")
            if avg and float(avg) > 0:
                order["average"] = float(avg)
                order["filled"]  = float(details.get("filled") or order.get("filled") or 0)
                return order
        except Exception:
            pass
        time.sleep(0.5)

    # Fallback نهائي
    px = fetch_price(sym)
    if px > 0:
        order["average"] = px
    return order

def _adjust_amount(sym: str, amount: float, market, enforce_min_cost: bool = True) -> float:
    """تصحيح الكمية حسب دقة السوق وحدوده."""
    # دقة الكمية
    try:
        amount = float(exchange.amount_to_precision(sym, amount))
    except Exception:
        pass

    if market is None:
        return max(0.0, amount)

    # حد أدنى للكمية
    try:
        min_amt = float(market.get("limits", {}).get("amount", {}).get("min") or 0.0)
        if min_amt and amount < min_amt:
            try:
                amount = float(exchange.amount_to_precision(sym, min_amt))
            except Exception:
                amount = min_amt
    except Exception:
        pass

    # حد أدنى للتكلفة (للشراء فقط)
    if enforce_min_cost:
        try:
            min_cost = float(market.get("limits", {}).get("cost", {}).get("min") or 0.0)
            if min_cost > 0:
                px = fetch_price(sym)
                if px > 0 and amount * px < min_cost:
                    needed = min_cost / px
                    try:
                        amount = float(exchange.amount_to_precision(sym, needed))
                    except Exception:
                        amount = needed
        except Exception:
            pass

    return max(0.0, amount)

# ================ fetch_symbol_filters — موثوقة =================
def fetch_symbol_filters(symbol: str) -> Dict[str, float]:
    """
    فلاتر الرمز من exchange.market():
    - stepSize, tickSize, minQty, minNotional
    النسخة الموثوقة التي تقرأ من OKX فعلياً.
    """
    sym = _fmt_symbol(symbol)
    try:
        m = exchange.market(sym)
    except Exception as e:
        print(f"[fetch_symbol_filters] market() error {sym}: {e}", flush=True)
        m = None

    step_size    = 1e-6
    tick_size    = 1e-6
    min_qty      = 0.0
    min_notional = 0.0

    if m:
        # stepSize
        try:
            lot_sz = float(m.get("info", {}).get("lotSz", 0) or 0)
            if lot_sz > 0:
                step_size = lot_sz
            else:
                dec = m.get("precision", {}).get("amount")
                step = _decimals_to_step(dec)
                if step:
                    step_size = step
        except Exception:
            pass

        # tickSize
        try:
            tick_sz = float(m.get("info", {}).get("tickSz", 0) or 0)
            if tick_sz > 0:
                tick_size = tick_sz
            else:
                pdec = m.get("precision", {}).get("price")
                step = _decimals_to_step(pdec)
                if step:
                    tick_size = step
        except Exception:
            pass

        # minQty
        try:
            min_qty = float(
                m.get("limits", {}).get("amount", {}).get("min")
                or m.get("info", {}).get("minSz")
                or 0.0
            )
        except Exception:
            min_qty = 0.0

        # minNotional
        try:
            min_cost = m.get("limits", {}).get("cost", {}).get("min")
            if min_cost is not None:
                min_notional = float(min_cost or 0.0)
            elif min_qty > 0:
                px = fetch_price(sym)
                min_notional = min_qty * px if px > 0 else 0.0
        except Exception:
            pass

    return {
        "stepSize":    float(max(1e-12, step_size)),
        "tickSize":    float(max(1e-12, tick_size)),
        "minQty":      float(max(0.0,   min_qty)),
        "minNotional": float(max(0.0,   min_notional)),
    }

# ================ أدوات إضافية =================
def list_okx_usdt_spot_symbols() -> List[str]:
    """قائمة أزواج SPOT/USDT بصيغة BTC/USDT."""
    try:
        r = requests.get(TICKERS_URL, timeout=12)
        r.raise_for_status()
        j = r.json()
        seen, out = set(), []
        for it in j.get("data", []):
            inst = str(it.get("instId", "")).upper()
            if inst.endswith("-USDT"):
                sym = inst.replace("-", "/")
                if sym not in seen:
                    out.append(sym)
                    seen.add(sym)
        return out
    except Exception as e:
        print(f"⚠️ list_okx_usdt_spot_symbols: {e}", flush=True)
        return []
