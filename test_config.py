from config import OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

def test_config():
    print("اختبار تحميل مفاتيح OKX والTelegram...")
    print(f"OKX_API_KEY: {OKX_API_KEY}")
    print(f"OKX_SECRET_KEY: {OKX_SECRET_KEY}")
    print(f"OKX_PASSPHRASE: {OKX_PASSPHRASE}")
    print(f"TELEGRAM_TOKEN: {TELEGRAM_TOKEN}")
    print(f"TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID}")

    # تحقق إذا أي مفتاح فارغ
    if not all([OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHRASE, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID]):
        print("❌ يوجد متغير فارغ! تحقق من ملف config.py")
    else:
        print("✅ جميع المفاتيح موجودة وجاهزة للاستخدام.")

if __name__ == "__main__":
    test_config()
