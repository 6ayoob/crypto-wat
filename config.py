
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
# 📈 الرموز — قائمة منقّحة (سيتم فلترتها تلقائياً عند التشغيل عبر OKX markets)
# ملاحظة: لا تعتمد على هذه القائمة فقط؛ okx_api سيزيل أي رمز غير مدعوم.
# ===============================
SYMBOLS = [
    # DeFi / Bluechips
    "AAVE/USDT", "UNI/USDT", "SUSHI/USDT", "COMP/USDT", "MKR/USDT",
    "SNX/USDT", "LDO/USDT", "GRT/USDT", "LINK/USDT",

    # Layer 1 / Majors
    "ETH/USDT", "SOL/USDT", "ADA/USDT", "AVAX/USDT", "NEAR/USDT",
    "ATOM/USDT", "DOT/USDT",

    # Gaming/Metaverse
    "MANA/USDT", "AXS/USDT", "ZENT/USTT",  # ← تأكد من "USDT" (لو خطأ مطبعي صحّحه)
    "CHZ/USDT", "ENJ/USDT", "GALA/USDT", "APE/USDT", "NMR/USDT", "AGLD/USDT"

    # Layer 2 / Infra
    "OP/USDT", "IMX/USDT", "ARB/USDT",  # أضفت ARB لأنها شائعة
    "ZIL/USDT", "ZRX/USDT", "SKL/USDT",

    # AI / Render / Web3
    "BAL/USDT",         # كان RENDER → RNDR
    "MERL/USDT",          # بدل FET (بعد الدمج، لو غير متاح سيتفلتر تلقائياً)
    "GLMR/USDT", "T/USDT", "BADGER/USDT", "PHA/USDT", "KNC/USDT", "BICO/USDT",

    # Meme/Trendy (أبقِ عددها محدوداً لأنها كثيرة الضوضاء)
    "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "BONK/USDT", "WIF/USDT",
    "ORDI/USDT", "FLOKI/USDT", "NOT/USDT"
]

# ===============================
# ⏱ إعدادات التداول
# ملاحظة: TIMEFRAME هنا "تراثية" للنسخ القديمة — الاستراتيجية الحالية تستخدم HTF/LTF داخل strategy.py
# ===============================
TIMEFRAME = "5m"                 # غير مستخدمة في المنطق الجديد، اتركها للتوافق فقط
TRADE_AMOUNT_USDT = 45           # قيمة الصفقة الافتراضية (يُمكن تفعيل sizing بالريسك من strategy.py)
MAX_OPEN_POSITIONS = 3           # الحد الأقصى للصفقات المفتوحة

# ===============================
# 🧮 الرسوم (round-trip) بالـ bps
# اضبطها على إجمالي الذهاب/الإياب. مثال: 16 = 0.16% إجمالي.
# ===============================
FEE_BPS_ROUNDTRIP = 16
