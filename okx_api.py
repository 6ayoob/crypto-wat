# okx_api.py — متوافق مع strategy.py
# المتطلبات: pip install ccxt
import os
import time
import ccxt

from config import API_KEY, SECRET_KEY, PASSPHRASE

exchange = ccxt.okx({
    "apiKey": API_KEY,
    "secret": SECRET_KEY,
    "password": PASSPHRASE,   # مهم في OKX
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
    },
})

# تفعيل الساندبوكس عند الحاجة
if os.getenv("OKX_TESTNET", "false").lower() in ("1", "true", "yes"):
    try:
        exchange.set_sandbox_mode(True)
    except Exception:
        pass

def _retry(times=3, delay=0.6):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last = None
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    time.sleep(delay * (1.5 ** i))
            raise last
        return wrapper
    return deco

def format_symbol(symbol: str) -> str:
    return symbol.replace("-", "/")

@_retry()
def fetch_balance(asset: str = "USDT") -> float:
    try:
        balances = exchange.fetch_balance()
        free = balances.get("free", {}) or {}
        return float(free.get(asset, 0.0) or 0.0)
    except Exception as e:
        print(f"❌ خطأ في جلب الرصيد لـ {asset}: {e}")
        return 0.0

@_retry()
def fetch_price(symbol: str) -> float:
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
    sym = format_symbol(symbol)
    try:
        data = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
        return data or []
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات الشموع لـ {sym}: {e}")
        return []

def _amount_to_precision(symbol_ccxt: str, amount: float) -> float:
    try:
        market = exchange.market(symbol_ccxt)
        amt = float(exchange.amount_to_precision(symbol_ccxt, amount))
        min_amt = float(market.get("limits", {}).get("amount", {}).get("min") or 0.0)
        if min_amt and amt < min_amt:
            amt = float(exchange.amount_to_precision(symbol_ccxt, min_amt))
        min_cost = float(market.get("limits", {}).get("cost", {}).get("min") or 0.0)
        if min_cost > 0:
            try:
                ticker = exchange.fetch_ticker(symbol_ccxt)
                last = float(ticker.get("last") or ticker.get("close") or ticker.get("ask") or 0)
                if last > 0 and amt * last < min_cost:
                    needed = min_cost / last
                    amt = float(exchange.amount_to_precision(symbol_ccxt, needed))
            except Exception:
                pass
        return max(0.0, amt)
    except Exception:
        return max(0.0, float(amount))

@_retry()
def place_market_order(symbol: str, side: str, amount: float, send_message=None):
    sym = format_symbol(symbol)
    side = (side or "").lower().strip()
    if side not in ("buy", "sell"):
        msg = f"⚠️ side غير صحيح ({side}) — يجب buy أو sell"
        print(msg);  send_message and send_message(msg)
        return None

    if amount is None or amount <= 0:
        msg = f"⚠️ كمية غير صالحة لأمر {side} على {symbol}"
        print(msg);  send_message and send_message(msg)
        return None

    adj_amount = _amount_to_precision(sym, float(amount))
    if adj_amount <= 0:
        msg = f"⚠️ الكمية بعد التسوية أصبحت صفر لـ {symbol} — تحقق من الحد الأدنى للسوق."
        print(msg);  send_message and send_message(msg)
        return None

    try:
        order = exchange.create_order(sym, type="market", side=side, amount=adj_amount)
        msg = f"✅ تم تنفيذ أمر {side.upper()} لـ {symbol} (كمية: {adj_amount})"
        print(msg);  send_message and send_message(msg)
        return order
    except Exception as e:
        msg = f"❌ فشل تنفيذ أمر السوق ({side}) لـ {symbol}: {str(e)}"
        print(msg);  send_message and send_message(msg)
        return None
