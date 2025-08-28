# okx_api.py — نسخة محسّنة ومتوافقة مع strategy.py (80/100)
# المتطلبات: pip install ccxt
import os
import time
import math
import ccxt

from config import API_KEY, SECRET_KEY, PASSPHRASE

# ============== تهيئة OKX ==============
exchange = ccxt.okx({
    "apiKey": API_KEY,
    "secret": SECRET_KEY,
    "password": PASSPHRASE,   # مهم في OKX
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
    },
})

# (اختياري) تفعيل الساندبوكس إذا أردت عبر متغير بيئة
# بعض البورصات يدعمها ccxt تلقائياً
if os.getenv("OKX_TESTNET", "false").lower() in ("1", "true", "yes"):
    try:
        exchange.set_sandbox_mode(True)
    except Exception:
        pass

# ============== أدوات مساعدة ==============
def _retry(times=3, delay=0.6):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last = None
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    # backoff بسيط
                    time.sleep(delay * (1.5 ** i))
            raise last
        return wrapper
    return deco

def format_symbol(symbol: str) -> str:
    """دعم 'BTC-USDT' و'BTC/USDT' بالتوحيد إلى الشكل القياسي لـ CCXT."""
    return symbol.replace("-", "/")

def _amount_to_precision(symbol_ccxt: str, amount: float) -> float:
    """تسوية الكمية حسب دقة السوق وحدوده."""
    try:
        market = exchange.market(symbol_ccxt)
        # دقة الكمية
        amt = float(exchange.amount_to_precision(symbol_ccxt, amount))
        # حد أدنى للكمية
        min_amt = float(market.get("limits", {}).get("amount", {}).get("min") or 0) or 0.0
        if min_amt and amt < min_amt:
            # إذا الكمية أصغر من الحد الأدنى نرفعها لأقرب حد مدعوم
            amt = float(exchange.amount_to_precision(symbol_ccxt, min_amt))
        # حد أدنى للتكلفة (اختياري إن توفّر)
        min_cost = float(market.get("limits", {}).get("cost", {}).get("min") or 0) or 0.0
        if min_cost > 0:
            try:
                ticker = exchange.fetch_ticker(symbol_ccxt)
                last = float(ticker.get("last") or ticker.get("close") or ticker.get("ask") or 0)
                if last > 0 and amt * last < min_cost:
                    # ارفع الكمية لتجاوز الحد الأدنى للتكلفة
                    needed = min_cost / last
                    amt = float(exchange.amount_to_precision(symbol_ccxt, needed))
            except Exception:
                pass
        return amt
    except Exception:
        # لو فشلنا بأي سبب، نرجع الكمية كما هي (قد يرفضها السيرفر إن كانت غير صالحة)
        return float(max(0.0, amount))

# ============== الدوال المطلوبة للاستراتيجية ==============
@_retry()
def fetch_balance(asset: str = "USDT") -> float:
    """
    يعيد الرصيد المتاح (free) للعملة المطلوبة كقيمة float.
    """
    try:
        balances = exchange.fetch_balance()
        free = balances.get("free", {}) or {}
        return float(free.get(asset, 0.0) or 0.0)
    except Exception as e:
        print(f"❌ خطأ في جلب الرصيد لـ {asset}: {e}")
        return 0.0

@_retry()
def fetch_price(symbol: str) -> float:
    """
    يعيد آخر سعر كـ float. يستخدم last ثم close/ask كبدائل.
    """
    sym = format_symbol(symbol)
    try:
        t = exchange.fetch_ticker(sym)
        price = t.get("last") or t.get("close") or t.get("ask") or t.get("bid")
        return float(price or 0.0)
    except Exception as e:
        print(f"❌ خطأ في جلب السعر الحالي لـ {sym}: {e}")
        return 0.0

@_retry()
def fetch_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 100):
    """
    يعيد قائمة OHLCV بالشكل: [[ts, open, high, low, close, volume], ...]
    """
    sym = format_symbol(symbol)
    try:
        data = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        return data or []
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات الشموع لـ {sym}: {e}")
        return []

@_retry()
def place_market_order(symbol: str, side: str, amount: float, send_message=None):
    """
    ينفذ أمر سوق Spot مع تسوية الكمية حسب دقة السوق وحدوده.
    يعيد كائن الطلب عند النجاح (truthy)، أو None عند الفشل.
    """
    sym = format_symbol(symbol)
    side = (side or "").lower().strip()
    if side not in ("buy", "sell"):
        msg = f"⚠️ side غير صحيح ({side}) — يجب buy أو sell"
        print(msg)
        if send_message:
            send_message(msg)
        return None

    if amount is None or amount <= 0:
        msg = f"⚠️ الكمية صفر أو غير صالحة عند محاولة تنفيذ أمر {side} لـ {symbol}"
        print(msg)
        if send_message:
            send_message(msg)
        return None

    # تسوية الكمية على قيود السوق
    adj_amount = _amount_to_precision(sym, float(amount))
    if adj_amount <= 0:
        msg = f"⚠️ الكمية بعد التسوية أصبحت صفر لـ {symbol} — تحقق من الحد الأدنى للسوق."
        print(msg)
        if send_message:
            send_message(msg)
        return None

    try:
        order = exchange.create_order(sym, type="market", side=side, amount=adj_amount)
        msg = f"✅ تم تنفيذ أمر {side.upper()} لـ {symbol} بنجاح (كمية: {adj_amount})"
        print(msg)
        if send_message:
            send_message(msg)
        return order
    except Exception as e:
        msg = f"❌ فشل تنفيذ أمر السوق ({side}) لـ {symbol}: {str(e)}"
        print(msg)
        if send_message:
            send_message(msg)
        return None
