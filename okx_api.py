# okx_api.py — متوافق مع strategy.py (محسّن)
# المتطلبات: pip install ccxt
import os
import time
import random
import traceback
import ccxt
from typing import Optional, Tuple, Dict, Any

from config import API_KEY, SECRET_KEY, PASSPHRASE

# ================= إعداد واجهة OKX ================
exchange = ccxt.okx({
    "apiKey": API_KEY,
    "secret": SECRET_KEY,
    "password": PASSPHRASE,   # مهم في OKX
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
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

# ================ أدوات مساعدة عامة ================
def _fmt_symbol(symbol: str) -> str:
    # يدعم صيغ مثل BTC-USDT, BTC_USDT → BTC/USDT
    return symbol.replace("-", "/").replace("_", "/").upper()

def _okx_error_hint(e: Exception) -> str:
    msg = str(e)
    # تلميحات لأخطاء OKX المعروفة
    if "50110" in msg or "IP" in msg and "whitelist" in msg.lower():
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
                    # لو خطأ حد المعدّل/شبكي، ننتظر Backoff + Jitter
                    sleep_s = min(max_delay, base_delay * (1.8 ** i)) + random.uniform(0, 0.2)
                    time.sleep(sleep_s)
            # بعد انتهاء المحاولات
            raise last_exc
        return wrapper
    return deco

# ================ كاش بسيط ================
_BAL_CACHE: Dict[str, Tuple[float, float]] = {}    # {asset: (ts, balance)}
_TICKER_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}  # {sym: (ts, ticker)}
CACHE_TTL_SEC_BAL = 3.0
CACHE_TTL_SEC_TICKER = 1.0

def _get_cached_bal(asset: str) -> Optional[float]:
    ts_bal = _BAL_CACHE.get(asset)
    if not ts_bal:
        return None
    ts, val = ts_bal
    if (time.time() - ts) <= CACHE_TTL_SEC_BAL:
        return val
    return None

def _set_cached_bal(asset: str, val: float):
    _BAL_CACHE[asset] = (time.time(), val)

def _get_cached_ticker(sym: str) -> Optional[Dict[str, Any]]:
    t = _TICKER_CACHE.get(sym)
    if not t:
        return None
    ts, data = t
    if (time.time() - ts) <= CACHE_TTL_SEC_TICKER:
        return data
    return None

def _set_cached_ticker(sym: str, data: Dict[str, Any]):
    _TICKER_CACHE[sym] = (time.time(), data)

# ================ تسوية الكمية/الحدود ================
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
            # نستخدم آخر سعر من الـ ticker (من الكاش أو الشبكة)
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
    """
    يعيد الرصيد (Free) للعملة المطلوبة. يستخدم كاش قصير لتخفيف الضغط.
    """
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
    """
    يعيد آخر سعر تداول/إغلاق/عرض/طلب — مع كاش قصير (1s).
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
        print(f"❌ خطأ في جلب السعر الحالي لـ {sym}: {e}")
        return 0.0

@_retry()
def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 100):
    """
    يعيد بيانات الشموع بصيغة CCXT: [[ts, open, high, low, close, volume], ...]
    NOTE: OKX يدعم حدودًا مختلفة حسب الإطار؛ 200–300 آمن عادةً.
    """
    sym = _fmt_symbol(symbol)
    # سقف معقول لتجنّب الرفض من OKX
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
        msg = f"⚠️ side غير صحيح ({side}) — يجب buy أو sell"
        _log(send_message, msg)
        return None

    if amount is None or amount <= 0:
        msg = f"⚠️ كمية غير صالحة لأمر {side} على {symbol}"
        _log(send_message, msg)
        return None

    adj_amount = _amount_to_precision(sym, float(amount))
    if adj_amount <= 0:
        msg = f"⚠️ الكمية بعد التسوية أصبحت صفر لـ {symbol} — تحقق من الحد الأدنى للسوق."
        _log(send_message, msg)
        return None

    try:
        order = exchange.create_order(sym, type="market", side=side, amount=adj_amount)
        _log(send_message, f"✅ تم تنفيذ أمر {side.upper()} لـ {symbol} (كمية: {adj_amount})")
        return order
    except Exception as e:
        hint = _okx_error_hint(e)
        if hint:
            _log(send_message, hint)
        _log(send_message, f"❌ فشل تنفيذ أمر السوق ({side}) لـ {symbol}: {str(e)}")
        # print(traceback.format_exc())  # فعّل عند الحاجة
        return None
