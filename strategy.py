import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from okx_api import fetch_ohlcv, fetch_price, place_market_order, fetch_balance
from config import TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS

# ===============================
# 📂 ملفات الصفقات
# ===============================
POSITIONS_DIR = "positions"
CLOSED_POSITIONS_FILE = "closed_positions.json"
RISK_STATE_FILE = "risk_state.json"

def ensure_dirs():
    os.makedirs(POSITIONS_DIR, exist_ok=True)

def get_position_filename(symbol):
    ensure_dirs()
    return f"{POSITIONS_DIR}/{symbol.replace('/', '_')}.json"

def load_position(symbol):
    try:
        file = get_position_filename(symbol)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ خطأ في قراءة الصفقة: {e}")
    return None

def save_position(symbol, position):
    try:
        file = get_position_filename(symbol)
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(position, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ خطأ في حفظ الصفقة: {e}")

def clear_position(symbol):
    try:
        file = get_position_filename(symbol)
        if os.path.exists(file):
            os.remove(file)
    except Exception as e:
        print(f"⚠️ خطأ في حذف الصفقة: {e}")

def count_open_positions():
    ensure_dirs()
    return len([f for f in os.listdir(POSITIONS_DIR) if f.endswith(".json")])

def load_closed_positions():
    try:
        if os.path.exists(CLOSED_POSITIONS_FILE):
            with open(CLOSED_POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ خطأ في قراءة الصفقات المغلقة: {e}")
    return []

def save_closed_positions(closed_positions):
    try:
        with open(CLOSED_POSITIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(closed_positions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ خطأ في حفظ الصفقات المغلقة: {e}")

# ===============================
# 📊 المؤشرات الفنية
# ===============================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().replace(0,1e-9)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(df, fast=12, slow=26, signal=9):
    df['ema_fast'] = ema(df['close'], fast)
    df['ema_slow'] = ema(df['close'], slow)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    return df

def calculate_indicators(df):
    df['ema9'] = ema(df['close'],9)
    df['ema21'] = ema(df['close'],21)
    df['ema50'] = ema(df['close'],50)
    df['rsi'] = rsi(df['close'],14)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df = macd(df)
    return df

# ===============================
# 🔎 دعم ومقاومة
# ===============================
def get_support_resistance(df, window=50):
    if len(df) < 5:
        return None, None
    df_prev = df.iloc[:-1]
    use_window = min(window,len(df_prev))
    resistance = df_prev['high'].rolling(use_window).max().iloc[-1]
    support = df_prev['low'].rolling(use_window).min().iloc[-1]
    return support,resistance

# ===============================
# ⚙️ إعدادات الاستراتيجية والحماية
# ===============================
SR_WINDOW = 50
RESISTANCE_BUFFER = 0.005
SUPPORT_BUFFER = 0.002
TRAILING_DISTANCE = 0.01
PARTIAL_FRACTION = 0.5

DAILY_MAX_LOSS_USDT = 50
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_MINUTES_AFTER_HALT = 120
MIN_USDT_5M_LIQUIDITY = 30000

MAX_TRADES_PER_DAY = 5

# ===============================
# 🛡️ إدارة حالة المخاطر اليومية
# ===============================
def _today_str():
    return datetime.utcnow().strftime("%Y-%m-%d")

def load_risk_state():
    state = {"date":_today_str(),"daily_pnl":0,"consecutive_losses":0,"trades_today":0,"blocked_until":None}
    try:
        if os.path.exists(RISK_STATE_FILE):
            with open(RISK_STATE_FILE,'r',encoding='utf-8') as f:
                data = json.load(f)
                if data.get("date") != _today_str(): return state
                return data
    except: pass
    return state

def save_risk_state(s):
    try:
        with open(RISK_STATE_FILE,'w',encoding='utf-8') as f:
            json.dump(s,f,indent=2,ensure_ascii=False)
    except: pass

def is_trading_blocked():
    s = load_risk_state()
    if s.get("blocked_until"):
        try:
            until = datetime.fromisoformat(s["blocked_until"])
            if datetime.utcnow() < until: return True, f"⏸️ التداول موقوف حتى {until.isoformat()}."
        except: pass
    if s.get("daily_pnl",0.0) <= -DAILY_MAX_LOSS_USDT: return True,"⛔ تم بلوغ حد الخسارة اليومية."
    if s.get("consecutive_losses",0) >= MAX_CONSECUTIVE_LOSSES: return True,"⛔ تم بلوغ حد الخسائر المتتالية."
    if s.get("trades_today",0) >= MAX_TRADES_PER_DAY: return True,f"⛔ تم بلوغ الحد الأقصى للصفقات اليوم ({MAX_TRADES_PER_DAY})."
    return False,""

def trigger_cooldown(reason="risk_halt"):
    s = load_risk_state()
    until = datetime.utcnow() + timedelta(minutes=COOLDOWN_MINUTES_AFTER_HALT)
    s["blocked_until"] = until.isoformat()
    save_risk_state(s)
    print(f"⏸️ تفعيل تهدئة حتى {until.isoformat()} ({reason}).")

def register_trade_result(total_pnl_usdt):
    s = load_risk_state()
    s["daily_pnl"] += total_pnl_usdt
    s["consecutive_losses"] = 0 if total_pnl_usdt>0 else s.get("consecutive_losses",0)+1
    save_risk_state(s)

# ===============================
# 🔎 فحص الإشارة
# ===============================
def check_signal(symbol):
    blocked,msg = is_trading_blocked()
    if blocked: print(msg); return None
    data = fetch_ohlcv(symbol,'5m',150)
    if not data: return None
    df = pd.DataFrame(data,columns=['timestamp','open','high','low','close','volume'])
    df = calculate_indicators(df)
    if len(df)<50: return None
    last = df.iloc[-1]; prev = df.iloc[-2]

    # فلتر الحجم والشمعة صاعدة
    if len(df['volume'])>=20:
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        if last['volume']<avg_vol or last['close']<=last['open']: return None
    # الاتجاه EMA50
    if last['close'] < last['ema50']: return None
    # RSI
    if not (50<last['rsi']<70): return None
    # الدعم والمقاومة
    support,resistance = get_support_resistance(df,SR_WINDOW)
    price = float(last['close'])
    if support and resistance:
        if price>=resistance*(1-RESISTANCE_BUFFER) or price<=support*(1+SUPPORT_BUFFER): return None
    # تقاطع EMA9 و EMA21
    if prev['ema9']<prev['ema21'] and last['ema9']>last['ema21'] and last['macd']>last['macd_signal']:
        return "buy"
    return None

# ===============================
# 🔎 تنفيذ الشراء
# ===============================
def execute_buy(symbol):
    blocked,msg = is_trading_blocked()
    if blocked: return None,msg
    if count_open_positions()>=MAX_OPEN_POSITIONS: return None,f"🚫 الحد الأقصى للصفقات المفتوحة."
    price = fetch_price(symbol)
    usdt_balance = fetch_balance('USDT')
    if usdt_balance<TRADE_AMOUNT_USDT: return None,f"🚫 رصيد USDT غير كافٍ."

    amount = TRADE_AMOUNT_USDT/price
    order = place_market_order(symbol,'buy',amount)
    if not order: return None,"⚠️ فشل تنفيذ الصفقة."

    # وقف خسارة من آخر 10 شمعات
    data = fetch_ohlcv(symbol,'5m',20)
    df = pd.DataFrame(data,columns=['timestamp','open','high','low','close','volume'])
    swing_low = df['low'].rolling(10).min().iloc[-2]
    stop_loss = float(swing_low)
    take_profit = price + (price-stop_loss)*2  # RR 1:2

    position = {
        "symbol": symbol,
        "amount": amount,
        "entry_price": price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "trailing_stop": price*(1-TRAILING_DISTANCE),
        "partial_done": False
    }
    save_position(symbol,position)

    # تحديث عدد الصفقات اليومية
    s = load_risk_state()
    s["trades_today"] += 1
    save_risk_state(s)

    return order,f"✅ تم شراء {symbol} بسعر {price:.8f} | TP:{take_profit:.8f} | SL:{stop_loss:.8f}"
# ===============================
# 🔧 إدارة الصفقات تلقائيًا
# ===============================
def manage_position(symbol):
    try:
        position = load_position(symbol)
        if not position:
            return False

        current_price = fetch_price(symbol)
        amount = position['amount']
        entry_price = position['entry_price']

        base_asset = symbol.split('/')[0]
        actual_balance = fetch_balance(base_asset)
        sell_amount = min(amount, actual_balance)

        closed = False

        # 🔹 Partial TP
        if not position.get("partial_done") and current_price >= entry_price + (position['take_profit']-entry_price)/2:
            partial_amount = sell_amount * PARTIAL_FRACTION
            order = place_market_order(symbol,'sell',partial_amount)
            if order:
                position['amount'] -= partial_amount
                position['partial_done'] = True
                save_position(symbol,position)
                closed = True
                pnl = (current_price - entry_price) * partial_amount
                register_trade_result(pnl)
                print(f"📌 Partial TP {symbol}: {pnl:.2f} USDT")
        
        # 🔹 Take Profit
        if current_price >= position['take_profit']:
            order = place_market_order(symbol,'sell',sell_amount)
            if order:
                pnl = (current_price - entry_price) * sell_amount
                close_trade(symbol,pnl)
                closed = True
                print(f"🎯 Take Profit {symbol}: {pnl:.2f} USDT")
        
        # 🔹 Stop Loss
        elif current_price <= position['stop_loss']:
            order = place_market_order(symbol,'sell',sell_amount)
            if order:
                pnl = (current_price - entry_price) * sell_amount
                close_trade(symbol,pnl)
                closed = True
                print(f"🛑 Stop Loss {symbol}: {pnl:.2f} USDT")
                # التحقق من الخسائر المتتالية
                s = load_risk_state()
                if s.get("consecutive_losses",0) >= MAX_CONSECUTIVE_LOSSES:
                    trigger_cooldown("max_consecutive_losses")
        
        # 🔹 Trailing Stop
        elif current_price > position['trailing_stop']/(1-TRAILING_DISTANCE):
            position['trailing_stop'] = current_price*(1-TRAILING_DISTANCE)
            save_position(symbol,position)
        
    except Exception as e:
        print(f"⚠️ خطأ في إدارة الصفقة لـ {symbol}: {e}")

    return closed

# ===============================
# 🔹 إغلاق الصفقة وتسجيلها
# ===============================
def close_trade(symbol,pnl):
    position = load_position(symbol)
    if not position: return
    closed_positions = load_closed_positions()
    closed_positions.append({
        "symbol": symbol,
        "entry_price": position['entry_price'],
        "exit_price": fetch_price(symbol),
        "amount": position['amount'],
        "profit": pnl,
        "closed_at": datetime.utcnow().isoformat()
    })
    save_closed_positions(closed_positions)
    register_trade_result(pnl)
    clear_position(symbol)

# ===============================
# 🔄 حلقة التشغيل التلقائي
# ===============================
def run_bot():
    while True:
        for symbol in SYMBOLS:
            signal = check_signal(symbol)
            if signal == "buy":
                order,msg = execute_buy(symbol)
                print(msg)
            manage_position(symbol)
        time.sleep(60)  # تكرار كل دقيقة

# ===============================
# 🔹 بدء البوت
# ===============================
if __name__ == "__main__":
    print("🚀 بدء تشغيل البوت Spot OKX")
    run_bot()
