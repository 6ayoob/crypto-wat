# config.py

API_KEY = "6e2d2b3f-636a-424a-a97e-5154e39e525a"
SECRET_KEY = "D4B9966385BEE5A7B7D8791BA5C0539F"
PASSPHRASE = "Ta123456&"

TELEGRAM_TOKEN = "8300868885:AAEx8Zxdkz9CRUHmjJ0vvn6L3kC2kOPCHuk"
TELEGRAM_CHAT_ID = "658712542"

# ✅ رموز OKX بصيغة صحيحة
SYMBOLS = [
    "CRV-USDT",    # Store of Value
    "GALA-USDT",   # Gaming
    "BNB-USDT",    # Exchange Chain
    "SOL-USDT",    # Layer 1
    "ADA-USDT",    # Layer 1
    "AVAX-USDT",   # Layer 1
    "ATOM-USDT",   # Interoperability
    "DOT-USDT",    # Parachains
    "PEPE-USDT",
    "LINK-USDT",   # Oracle
    "UNI-USDT",    # DeFi
    "AAVE-USDT",   # DeFi Lending
    "SUSHI-USDT",  # DEX
    "LDO-USDT",    # Staking
    "INJ-USDT",    # DeFi Trading
    "XRP-USDT",   
    "FET-USDT",    # AI
    "APE-USDT",    # Metaverse
    "TIA-USDT",    # Modular Blockchain
    "OP-USDT",     # Optimism (L2)
]

# الإطار الزمني للشموع
TIMEFRAME = "5m"  # 1 دقيقة (مضاربات سريعة)

# إعدادات التداول
TRADE_AMOUNT_USDT = 15   # مبلغ التداول لكل صفقة بالدولار
STOP_LOSS_PCT = 0.02    # وقف خسارة 2%
TAKE_PROFIT_PCT = 0.04  # جني ربح 4%

MAX_OPEN_POSITIONS = 4  # الحد الأقصى لعدد الصفقات المفتوحة في نفس الوقت
