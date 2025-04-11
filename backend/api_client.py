import os
from dotenv import load_dotenv
from tinkoff.invest import Client, CandleInterval
from datetime import datetime, timedelta, timezone

# Загрузка .env из корня проекта
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

class TinkoffApiClient:
    def __init__(self, token: str = None):
        self.token = token or os.getenv("TINKOFF_TOKEN")
    
    def get_candles(self, figi: str, interval: str = "1min", days: int = 1):
        interval_map = {
            "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
            "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
            "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
            "hour": CandleInterval.CANDLE_INTERVAL_HOUR,
            "day": CandleInterval.CANDLE_INTERVAL_DAY
        }
        
        with Client(self.token) as client:
            now = datetime.now(timezone.utc)
            from_time = now - timedelta(days=days)
            
            resp = client.market_data.get_candles(
                instrument_id=figi,
                interval=interval_map[interval],
                from_=from_time,
                to=now
            )
            return resp.candles

# Для тестирования
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    client = TinkoffApiClient()
    print("Тестирование получения минутных данных...")
    candles = client.get_candles("BBG004730N88", "1min", 1)
    print(f"Получено {len(candles)} минутных свечей")