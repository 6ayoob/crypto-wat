
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
# 📈 الرموز
# (سيتم فلترتها تلقائياً في وقت التشغيل للتأكد أنها مدعومة على OKX)
# ===============================
SYMBOLS = [
  # DeFi (10)
  "AAVE/USDT", "UNI/USDT", "SUSHI/USDT", "COMP/USDT", "MKR/USDT",
  "SNX/USDT", "CRV/USDT", "LDO/USDT", "GRT/USDT", "LINK/USDT",
  # Layer 1 (10)
  "ETH/USDT", "SOL/USDT", "ADA/USDT", "AVAX/USDT", "NEAR/USDT",
  "ALGO/USDT", "ATOM/USDT", "DOT/USDT", "BNB/USDT", "FET/USDT",
  # Gaming/Metaverse (8)
  "MANA/USDT", "AXS/USDT", "SAND/USDT", "CHZ/USDT", "ENJ/USDT",
  "GALA/USDT", "APE/USDT", "ILV/USDT",
  # Layer 2 (5)
  "OP/USDT", "IMX/USDT", "LUNA/USDT", "ZIL/USDT", "ZRX/USDT", "SKL/USDT",
  # Meme Coins (5)
  "PEPE/USDT", "DOGE/USDT", "SHIB/USDT", "PUMP/USDT", "MEMEFI/USDT",
  # Stable / Oracles / Infra (10)
  "USDC/USDT", "DAI/USDT", "BAND/USDT", "API3/USDT", "AVAX/USDT",
  "LINK/USDT", "RSR/USDT", "UMA/USDT", "KNC/USDT", "BICO/USDT",
  # AI / Web3 / Others (10)
  "RENDER/USDT", "AIXBT/USDT", "VRA/USDT", "GLMR/USDT", "T/USDT",
  "PSTAKE/USDT", "BADGER/USDT", "PHA/USDT", "NC/USDT", "BOME/USDT",
  # رموز ميم من OKX
  "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "PENGU/USDT", 
  "BONK/USDT", "TRUMP/USDT", "FLOKI/USDT", "POLYDOGE/USDT",
  "WIF/USDT", "TURBO/USDT", "NOT/USDT", "ORDI/USDT",
  "DEGEN/USDT", "MEME/USDT", "DOGS/USDT", "VINE/USDT",
  "CAT/USDT", "ELON/USDT",
]

# ===============================
# ⏱ إعدادات التداول
# ===============================
TIMEFRAME = "5m"
TRADE_AMOUNT_USDT = 45         # ← تم الرفع إلى 45$
MAX_OPEN_POSITIONS = 3         # ← 3 كحد أقصى للصفقات المفتوحة

# ===============================
# 🧮 الرسوم (round-trip) بالـ bps
# ===============================
FEE_BPS_ROUNDTRIP = 8          # 0.08% ذهاب/إياب (عدّلها حسب حسابك)
