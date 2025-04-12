import sys
from pathlib import Path
from datetime import datetime, timedelta
import tkinter as tk
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.widgets import SpanSelector
import numpy as np

sys.path.append(str(Path(__file__).parent))

from backend.api_client import TinkoffApiClient
from backend.data_processor import DataProcessor

class StockMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Stock Monitor")
        self.geometry("1400x900")
        
        self.api = TinkoffApiClient()
        self.processor = DataProcessor()
        
        # Поддерживаемые интервалы с максимальными периодами
        self.period_settings = {
            "1 день": {"interval": "1min", "days": 1},
            "3 дня": {"interval": "5min", "days": 3},
            "1 неделя": {"interval": "15min", "days": 7},
            # Для периодов больше недели используем дневные данные через day=1
            "1 месяц": {"interval": "day", "days": 30},
            "3 месяца": {"interval": "day", "days": 90},
            "1 год": {"interval": "day", "days": 365}
        }
        
        self.current_data = None
        self.original_limits = None
        
        self.setup_ui()
        self.load_and_plot_data()
    
    def setup_ui(self):
        """Настройка интерфейса"""
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.figi_entry = ctk.CTkEntry(control_frame, width=300)
        self.figi_entry.pack(side=tk.LEFT, padx=5)
        self.figi_entry.insert(0, "BBG004730N88")  # SBER
        
        ctk.CTkButton(
            control_frame,
            text="Обновить",
            command=self.load_and_plot_data
        ).pack(side=tk.LEFT, padx=5)
        
        self.period_var = ctk.StringVar(value="1 день")
        ctk.CTkOptionMenu(
            control_frame,
            values=list(self.period_settings.keys()),
            variable=self.period_var,
            command=lambda _: self.load_and_plot_data()
        ).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(
            control_frame,
            text="Сбросить zoom",
            command=self.reset_zoom,
            fg_color="gray"
        ).pack(side=tk.LEFT, padx=5)
        
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.span = SpanSelector(
            self.ax,
            self.on_zoom_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='yellow'),
            interactive=True
        )
    
    def convert_quotation_to_float(self, quotation):
        """Конвертирует объект Quotation в float"""
        return float(quotation.units) + float(quotation.nano) / 1e9
    
    def load_and_plot_data(self):
        """Загрузка данных с учетом ограничений API"""
        figi = self.figi_entry.get().strip()
        if not figi:
            return
            
        try:
            period = self.period_var.get()
            settings = self.period_settings[period]
            
            # Получаем данные
            candles = self.api.get_candles(
                figi=figi,
                interval=settings["interval"],
                days=settings["days"]
            )
            
            if not candles:
                print("Нет данных для отображения")
                return
                
            # Обрабатываем данные
            timestamps = np.array([candle.time for candle in candles])
            prices = np.array([self.convert_quotation_to_float(candle.close) for candle in candles])
            
            self.current_data = (timestamps, prices)
            
            # Очищаем и рисуем график
            self.ax.clear()
            self.ax.plot(timestamps, prices, 'b-', linewidth=1)
            
            # Настройка осей
            self.configure_axes(settings["interval"])
            
            self.ax.set_title(f"{figi} ({period})")
            self.ax.grid(True, alpha=0.3)
            
            self.original_limits = (self.ax.get_xlim(), self.ax.get_ylim())
            self.canvas.draw()
            
        except Exception as e:
            print(f"Ошибка: {e}")
    
    def configure_axes(self, interval):
        """Настройка осей в зависимости от интервала"""
        locator_map = {
            "1min": mdates.HourLocator(interval=1),
            "5min": mdates.HourLocator(interval=1),
            "15min": mdates.DayLocator(interval=1),
            "day": mdates.DayLocator(interval=7)  # Для дневных данных
        }
        
        formatter_map = {
            "1min": mdates.DateFormatter('%H:%M'),
            "5min": mdates.DateFormatter('%H:%M'),
            "15min": mdates.DateFormatter('%Y-%m-%d'),
            "day": mdates.DateFormatter('%Y-%m-%d')  # Для дневных данных
        }
        
        self.ax.xaxis.set_major_locator(locator_map.get(interval))
        self.ax.xaxis.set_major_formatter(formatter_map.get(interval))
        self.fig.autofmt_xdate()
    
    def on_zoom_select(self, xmin, xmax):
        """Обработчик зума"""
        if self.current_data is None:
            return
            
        timestamps, prices = self.current_data
        
        xmin_dt = mdates.num2date(xmin)
        xmax_dt = mdates.num2date(xmax)
        
        mask = (timestamps >= xmin_dt) & (timestamps <= xmax_dt)
        visible_data = prices[mask]
        
        if len(visible_data) > 0:
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(
                np.min(visible_data) * 0.99,
                np.max(visible_data) * 1.01
            )
            self.canvas.draw_idle()
    
    def reset_zoom(self):
        """Сброс зума"""
        if self.original_limits:
            self.ax.set_xlim(self.original_limits[0])
            self.ax.set_ylim(self.original_limits[1])
            self.canvas.draw_idle()

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = StockMonitorApp()
    app.mainloop()