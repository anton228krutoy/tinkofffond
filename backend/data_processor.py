from datetime import datetime
from typing import List, Tuple

class DataProcessor:
    @staticmethod
    def process_candles(candles) -> Tuple[List[float], List[datetime]]:
        prices = []
        timestamps = []
        for candle in candles:
            price = float(candle.close.units) + candle.close.nano / 1e9
            prices.append(price)
            timestamps.append(candle.time.replace(tzinfo=None))
        return prices, timestamps

    @staticmethod
    def calculate_sma(prices: List[float], window: int) -> List[float]:
        if len(prices) < window:
            return []
        return [sum(prices[i:i+window])/window for i in range(len(prices)-window+1)]