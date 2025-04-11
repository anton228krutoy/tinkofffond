import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TOKEN = os.getenv("TINKOFF_TOKEN")
    DEFAULT_FIGI = "BBG004730N88"  # FIGI Сбербанка