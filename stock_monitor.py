import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, TextBox, CheckButtons
from tinkoff.invest import Client, CandleInterval
from datetime import datetime, timedelta, timezone

# Загрузка токена из .env
load_dotenv()
TOKEN = os.getenv("TINKOFF_TOKEN")

class StockMonitor:
    def __init__(self):
        self.figi = "BBG004730N88"  # FIGI Сбербанка
        self.interval = CandleInterval.CANDLE_INTERVAL_1_MIN
        self.days = 1
        self.prices = []
        self.timestamps = []
        self.show_sma = False
        self.sma_window = 20
        
        # Создаем фигуру с GridSpec
        self.fig = plt.figure(figsize=(14, 9), facecolor='#f0f0f0')
        gs = GridSpec(3, 3, height_ratios=[8, 1, 1], width_ratios=[1, 1, 1])
        
        # Основной график
        self.ax = self.fig.add_subplot(gs[0, :])
        self.ax.set_facecolor('#f8f8f8')
        
        # Панель управления
        self.control_panel = self.fig.add_subplot(gs[1:, :])
        self.control_panel.axis('off')
        
        self.setup_ui()
        self.update_data()
        
    def setup_ui(self):
        """Настройка пользовательского интерфейса с GridSpec"""
        # Поле для FIGI
        figi_box = plt.axes([0.1, 0.12, 0.25, 0.05])
        self.figi_input = TextBox(figi_box, 'FIGI:', initial=self.figi, color='white', hovercolor='#e0e0e0')
        
        # Поле для дней
        days_box = plt.axes([0.4, 0.12, 0.1, 0.05])
        self.days_input = TextBox(days_box, 'Дни:', initial=str(self.days), color='white', hovercolor='#e0e0e0')
        
        # Поле для SMA
        sma_box = plt.axes([0.55, 0.12, 0.1, 0.05])
        self.sma_input = TextBox(sma_box, 'SMA:', initial=str(self.sma_window), color='white', hovercolor='#e0e0e0')
        
        # Чекбокс для SMA
        sma_check = plt.axes([0.68, 0.12, 0.1, 0.05])
        self.sma_check = CheckButtons(sma_check, ['Показывать SMA'], [self.show_sma])
        self.sma_check.on_clicked(self.toggle_sma)
        
        # Кнопка обновления
        btn_box = plt.axes([0.82, 0.12, 0.15, 0.05])
        self.btn_update = Button(btn_box, 'Обновить данные', color='#4caf50', hovercolor='#66bb6a')
        self.btn_update.on_clicked(self.on_update_clicked)
        
        # Настройка стилей
        for text_box in [self.figi_input, self.days_input, self.sma_input]:
            text_box.label.set_color('#333333')
            text_box.text_disp.set_color('#333333')

    def toggle_sma(self, label):
        """Переключение отображения SMA"""
        self.show_sma = not self.show_sma
        self.draw_plot()
    
    def fetch_data(self):
        """Получение данных с API Tinkoff"""
        with Client(TOKEN) as client:
            now = datetime.now(timezone.utc)
            from_time = now - timedelta(days=self.days)
            
            resp = client.market_data.get_candles(
                instrument_id=self.figi,
                interval=self.interval,
                from_=from_time,
                to=now
            )
            
            self.prices = [
                float(candle.close.units) + candle.close.nano / 1e9
                for candle in resp.candles
            ]
            self.timestamps = [candle.time.replace(tzinfo=None) for candle in resp.candles]
            
            # Рассчитываем SMA если нужно
            if self.show_sma and len(self.prices) >= self.sma_window:
                self.sma = np.convolve(
                    self.prices, 
                    np.ones(self.sma_window)/self.sma_window, 
                    mode='valid'
                )
    
    def on_update_clicked(self, event):
        """Обработчик клика кнопки обновления"""
        self.figi = self.figi_input.text
        self.days = int(self.days_input.text)
        self.sma_window = int(self.sma_input.text)
        self.update_data()
    
    def update_data(self):
        """Обновление данных и графика"""
        try:
            self.fetch_data()
            self.draw_plot()
        except Exception as e:
            print(f"Ошибка: {e}")
    
    def draw_plot(self):
        """Отрисовка графика с улучшенным стилем"""
        self.ax.clear()
        
        if not self.prices:
            return
            
        # Основной график цены
        self.ax.plot(
            self.timestamps, 
            self.prices, 
            'b-', 
            linewidth=1.5,
            label="Цена закрытия",
            alpha=0.8
        )
        
        # SMA если включено
        if self.show_sma and hasattr(self, 'sma'):
            self.ax.plot(
                self.timestamps[self.sma_window-1:], 
                self.sma, 
                'r--', 
                linewidth=1.2,
                label=f"SMA {self.sma_window}"
            )
        
        # Настройки графика
        self.ax.set_title(
            f"Акции {self.figi} | {self.days} дней | {self.interval.name.replace('CANDLE_INTERVAL_', '')}",
            pad=20,
            fontsize=12,
            fontweight='bold'
        )
        
        self.ax.set_xlabel("Дата и время", labelpad=10)
        self.ax.set_ylabel("Цена, руб", labelpad=10)
        self.ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
        self.ax.legend(loc='upper left')
        
        # Форматирование дат
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        
        # Автомасштабирование с небольшими отступами
        y_min, y_max = min(self.prices), max(self.prices)
        self.ax.set_ylim(y_min * 0.99, y_max * 1.01)
        
        # Обновляем график
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    print("Скрипт начал выполнение...")
    monitor = StockMonitor()
    plt.tight_layout()
    plt.show()