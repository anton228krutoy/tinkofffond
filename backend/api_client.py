from tinkoff.invest import Client
import os
from dotenv import load_dotenv

load_dotenv()

class TinkoffApiClient:
    def __init__(self):
        self.token = os.getenv("TINKOFF_TOKEN")
    
    def get_candles(self, figi, interval, days):
        with Client(self.token) as client:
            from_time = ...  # добавьте логику дат
            return client.market_data.get_candles(
                instrument_id=figi,
                interval=interval,
                from_=from_time,
                to=datetime.now()
            ).candles