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
from lstm_predictor import LSTMPredictor  # Импорт модуля LSTM

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
        
        self.period_settings = {
            "1 день": {"interval": "1min", "days": 1},
            "3 дня": {"interval": "5min", "days": 3},
            "1 неделя": {"interval": "15min", "days": 7},
            "1 месяц": {"interval": "day", "days": 30},
            "3 месяца": {"interval": "day", "days": 90},
            "1 год": {"interval": "day", "days": 365}
        }
        
        self.current_data = None
        self.original_limits = None
        self.current_figi = None
        self.current_period = None
        
        self.setup_ui()
        self.load_and_plot_data()

        self.selected_range = None  # Для хранения выбранного диапазона
        self.lstm_predictor = None  # Объект LSTM

    
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
        
        # Добавляем аннотацию для отображения цены и времени
        self.price_annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->")
        )
        self.price_annotation.set_visible(False)

        # Добавляем кнопку для прогноза LSTM
        self.predict_button = ctk.CTkButton(
            control_frame,
            text="Прогноз LSTM",
            command=self.activate_range_selection,
            fg_color="green"
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Подключаем обработчики событий мыши
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("figure_leave_event", self.on_mouse_leave)
    
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
            self.current_figi = figi
            self.current_period = period
            
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


    def activate_range_selection(self):
        """Активирует выбор диапазона для прогноза."""
        self.predict_button.configure(text="Выберите диапазон на графике", fg_color="orange")
        self.span = SpanSelector(
            self.ax,
            self.on_range_selected,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red')
        )

    def on_range_selected(self, xmin, xmax):
        """Обрабатывает выбранный диапазон и строит прогноз."""
        if self.current_data is None:
            return

        timestamps, prices = self.current_data
        xmin_dt = mdates.num2date(xmin)
        xmax_dt = mdates.num2date(xmax)

        # Находим индексы выбранного диапазона
        mask = (timestamps >= xmin_dt) & (timestamps <= xmax_dt)
        selected_indices = np.where(mask)[0]

        if len(selected_indices) == 0:
            return

        start_idx, end_idx = selected_indices[0], selected_indices[-1]
        self.selected_range = (start_idx, end_idx)

        # Инициализируем и обучаем LSTM на всех данных до выбранного диапазона
        self.lstm_predictor = LSTMPredictor(prices[:end_idx].reshape(-1, 1))
        self.lstm_predictor.train(epochs=20)

        # Прогнозируем и отображаем
        pred_prices = self.lstm_predictor.predict(len(prices[start_idx:end_idx]))

        # Отображаем прогноз на графике
        self.ax.plot(timestamps[start_idx:end_idx], pred_prices, 'r--', label='LSTM Прогноз')
        
        # Возвращаем кнопку в исходное состояние
        self.predict_button.configure(text="Прогноз LSTM", fg_color="green")

    # ... (остальные методы класса)
    
    def configure_axes(self, interval):
        """Настройка осей в зависимости от интервала"""
        locator_map = {
            "1min": mdates.AutoDateLocator(),
            "5min": mdates.AutoDateLocator(),
            "15min": mdates.AutoDateLocator(),
            "day": mdates.AutoDateLocator()
        }
        
        formatter_map = {
            "1min": mdates.ConciseDateFormatter(locator_map[interval]),
            "5min": mdates.ConciseDateFormatter(locator_map[interval]),
            "15min": mdates.ConciseDateFormatter(locator_map[interval]),
            "day": mdates.ConciseDateFormatter(locator_map[interval])
        }
        
        self.ax.xaxis.set_major_locator(locator_map[interval])
        self.ax.xaxis.set_major_formatter(formatter_map[interval])
        self.fig.autofmt_xdate()
        
        # Настройка формата цены
        self.ax.yaxis.set_major_formatter('{x:,.2f}')
    
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
            # Добавляем небольшой отступ по y для лучшего отображения
            y_min = np.min(visible_data)
            y_max = np.max(visible_data)
            y_padding = (y_max - y_min) * 0.05  # 5% от диапазона
            
            self.ax.set_ylim(
                y_min - y_padding,
                y_max + y_padding
            )
            self.canvas.draw_idle()
    
    def reset_zoom(self):
        """Сброс зума"""
        if self.original_limits:
            self.ax.set_xlim(self.original_limits[0])
            self.ax.set_ylim(self.original_limits[1])
            self.canvas.draw_idle()
    
    def on_mouse_move(self, event):
        """Обработчик движения мыши для отображения цены и времени"""
        if event.inaxes != self.ax or self.current_data is None:
            self.price_annotation.set_visible(False)
            self.canvas.draw_idle()
            return
            
        timestamps, prices = self.current_data
        
        # Находим ближайшую точку данных
        xdata = mdates.num2date(event.xdata)
        idx = np.argmin(np.abs(timestamps - xdata))
        
        if 0 <= idx < len(prices):
            price = prices[idx]
            timestamp = timestamps[idx]
            
            # Форматируем время в зависимости от интервала
            if self.current_period in ["1 месяц", "3 месяца", "1 год"]:
                time_str = timestamp.strftime('%Y-%m-%d')
            else:
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            self.price_annotation.xy = (timestamp, price)
            self.price_annotation.set_text(f"{time_str}\nЦена: {price:.2f}")
            self.price_annotation.set_visible(True)
            self.canvas.draw_idle()
    
    def on_mouse_leave(self, event):
        """Скрываем аннотацию при выходе за пределы графика"""
        self.price_annotation.set_visible(False)
        self.canvas.draw_idle()

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = StockMonitorApp()
    app.mainloop()