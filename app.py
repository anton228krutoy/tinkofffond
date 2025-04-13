
#в1
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


'''
#в2
import sys
from pathlib import Path
from datetime import timedelta
import tkinter as tk
import customtkinter as ctk
import numpy as np
import matplotlib.dates as mdates
from lstm_predictor import LSTMPredictor
from plot_manager import PlotManager

sys.path.append(str(Path(__file__).parent))

from backend.api_client import TinkoffApiClient

class StockMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Stock Monitor")
        self.geometry("1400x900")
        self.api = TinkoffApiClient()
        
        self.period_settings = {
            "1 день": {"interval": "1min", "days": 1},
            "3 дня": {"interval": "5min", "days": 3},
            "1 неделя": {"interval": "15min", "days": 7},
            "1 месяц": {"interval": "day", "days": 30},
            "3 месяца": {"interval": "day", "days": 90},
            "1 год": {"interval": "day", "days": 365}
        }
        
        self.current_figi = None
        self.current_period = None
        self.lstm_predictor = None
        
        self.setup_ui()
        self.load_and_plot_data()

    def setup_ui(self):
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
        
        self.plot_manager = PlotManager(self.main_frame)
        
        self.predict_button = ctk.CTkButton(
            control_frame,
            text="Прогноз LSTM",
            command=self.activate_range_selection,
            fg_color="green"
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        self.plot_manager.canvas.mpl_connect(
            "motion_notify_event", self.plot_manager.on_mouse_move)
        self.plot_manager.canvas.mpl_connect(
            "figure_leave_event", self.plot_manager.on_mouse_leave)

    def load_and_plot_data(self):
        figi = self.figi_entry.get().strip()
        if not figi: return
        
        try:
            period = self.period_var.get()
            settings = self.period_settings[period]
            
            candles = self.api.get_candles(
                figi=figi,
                interval=settings["interval"],
                days=settings["days"]
            )
            
            if not candles:
                print("Нет данных")
                return
                
            timestamps = np.array([candle.time for candle in candles])
            prices = np.array([float(candle.close.units) + 
                             candle.close.nano/1e9 for candle in candles])
            
            self.current_figi = figi
            self.current_period = period
            
            self.plot_manager.plot_data(
                timestamps, prices, 
                f"{figi} ({period})", 
                settings["interval"]
            )
            
        except Exception as e:
            print(f"Ошибка: {e}")

    def activate_range_selection(self):
        self.predict_button.configure(
            text="Выберите диапазон для прогноза", 
            fg_color="orange")
        self.plot_manager.setup_range_selection(self.on_range_selected)

    def on_range_selected(self, xmin, xmax):
        if not hasattr(self.plot_manager, 'current_data'):
            return

        timestamps, prices = self.plot_manager.current_data
        xmin_dt = mdates.num2date(xmin)
        xmax_dt = mdates.num2date(xmax)

        mask = (timestamps >= xmin_dt) & (timestamps <= xmax_dt)
        target_indices = np.where(mask)[0]
        
        if len(target_indices) < 10:
            print("Слишком короткий диапазон")
            return

        train_data = prices[:target_indices[0]].reshape(-1, 1)
        
        try:
            self.lstm_predictor = LSTMPredictor(train_data, look_back=60)
            self.lstm_predictor.train(epochs=100, patience=15)
            
            pred_prices = self.lstm_predictor.predict(len(target_indices))
            true_prices = prices[target_indices]
            
            self.plot_manager.ax.clear()
            self.plot_manager.ax.plot(
                timestamps[:target_indices[0]], 
                prices[:target_indices[0]], 
                'b-', label='Исторические данные')
            self.plot_manager.ax.plot(
                timestamps[target_indices],
                true_prices,
                'g-', label='Реальные значения')
            self.plot_manager.ax.plot(
                timestamps[target_indices],
                pred_prices,
                'r--', linewidth=2, label='Прогноз LSTM')
            
            mse = np.mean((pred_prices - true_prices)**2)
            self.plot_manager.ax.set_title(
                f"Прогноз vs Реальность (MSE: {mse:.2f})")
            self.plot_manager.ax.legend()
            self.plot_manager.canvas.draw()
            
        except Exception as e:
            print(f"Ошибка: {e}")
        finally:
           self.predict_button.configure(text="Прогноз LSTM", fg_color="green")
           self.plot_manager.reset_zoom_selector()  # Возвращаем зум!

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = StockMonitorApp()
    app.mainloop()
'''

'''
#в3
import sys
from pathlib import Path
import tkinter as tk
import customtkinter as ctk
import numpy as np
import matplotlib.dates as mdates
from lstm_predictor import LSTMPredictor
from plot_manager import PlotManager

sys.path.append(str(Path(__file__).parent))

from backend.api_client import TinkoffApiClient

class StockMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Stock Monitor")
        self.geometry("1400x900")
        
        self.api = TinkoffApiClient()
        self.period_settings = {
            "1 день": {"interval": "1min", "days": 1},
            "3 дня": {"interval": "5min", "days": 3},
            "1 неделя": {"interval": "15min", "days": 7},
            "1 месяц": {"interval": "day", "days": 30},
            "3 месяца": {"interval": "day", "days": 90},
            "1 год": {"interval": "day", "days": 365}
        }
        
        self.interval_to_lookback = {
            "1min": 120,   # 2 часа для минутных данных
            "5min": 72,    # 6 часов для 5-минутных
            "15min": 64,   # 16 часов для 15-минутных
            "day": 30      # 30 дней для дневных
        }
        
        self.setup_ui()
        self.load_and_plot_data()
    
    def setup_ui(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Поле ввода FIGI
        self.figi_entry = ctk.CTkEntry(control_frame, width=300)
        self.figi_entry.pack(side=tk.LEFT, padx=5)
        self.figi_entry.insert(0, "BBG004730N88")  # SBER
        
        # Кнопка обновления
        ctk.CTkButton(
            control_frame, text="Обновить",
            command=self.load_and_plot_data
        ).pack(side=tk.LEFT, padx=5)
        
        # Выбор периода
        self.period_var = ctk.StringVar(value="1 день")
        ctk.CTkOptionMenu(
            control_frame,
            values=list(self.period_settings.keys()),
            variable=self.period_var,
            command=lambda _: self.load_and_plot_data()
        ).pack(side=tk.LEFT, padx=5)
        
        # Кнопка сброса зума
        ctk.CTkButton(
            control_frame, text="Сбросить zoom",
            command=lambda: self.plot_manager.reset_zoom(),
            fg_color="gray"
        ).pack(side=tk.LEFT, padx=5)
        
        # Кнопка прогноза LSTM
        self.predict_button = ctk.CTkButton(
            control_frame, text="Прогноз LSTM",
            command=self.activate_range_selection,
            fg_color="green"
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Статус-лейбл
        self.status_label = ctk.CTkLabel(control_frame, text="")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Инициализация графика
        self.plot_manager = PlotManager(self.main_frame)
        self.plot_manager.canvas.mpl_connect(
            "motion_notify_event", self.plot_manager.on_mouse_move)
        self.plot_manager.canvas.mpl_connect(
            "figure_leave_event", self.plot_manager.on_mouse_leave)
    
    def load_and_plot_data(self):
        """Загрузка и отображение данных"""
        figi = self.figi_entry.get().strip()
        if not figi:
            self.show_status("Введите FIGI", "red")
            return
            
        period = self.period_var.get()
        settings = self.period_settings[period]
        
        try:
            self.show_status("Загрузка данных...", "blue")
            self.update()  # Принудительное обновление интерфейса
            
            candles = self.api.get_candles(
                figi=figi,
                interval=settings["interval"],
                days=settings["days"]
            )
            
            if not candles:
                self.show_status("Нет данных для отображения", "red")
                return
                
            timestamps = np.array([candle.time for candle in candles])
            prices = np.array([float(candle.close.units) + 
                             candle.close.nano/1e9 for candle in candles])
            
            self.plot_manager.plot_data(
                timestamps, prices,
                f"{figi} ({period}, интервал: {settings['interval']})",
                settings["interval"]
            )
            self.show_status("Данные загружены", "green")
            
        except Exception as e:
            self.show_status(f"Ошибка: {e}", "red")
    
    def show_status(self, message, color):
        """Отображение статусного сообщения"""
        self.status_label.configure(text=message, text_color=color)
        self.update()
    
    def activate_range_selection(self):
        """Активация выбора диапазона для прогноза"""
        self.predict_button.configure(
            text="Выберите диапазон", 
            fg_color="orange",
            state="disabled"
        )
        self.show_status("Выделите область для прогноза", "blue")
        self.plot_manager.setup_range_selection(self.on_range_selected)
    
    def on_range_selected(self, xmin, xmax):
        """Обработка выбранного диапазона"""
        self.predict_button.configure(
            text="Обработка...", 
            fg_color="blue",
            state="disabled"
        )
        self.show_status("Анализ данных...", "blue")
        self.update()  # Принудительное обновление
        
        try:
            if not hasattr(self.plot_manager, 'current_data'):
                raise ValueError("Нет данных для анализа")

            timestamps, prices = self.plot_manager.current_data
            xmin_dt = mdates.num2date(xmin)
            xmax_dt = mdates.num2date(xmax)

            mask = (timestamps >= xmin_dt) & (timestamps <= xmax_dt)
            target_indices = np.where(mask)[0]
            
            if len(target_indices) < 10:
                raise ValueError("Минимум 10 точек для прогноза")

            period = self.period_var.get()
            interval = self.period_settings[period]["interval"]
            look_back = self.interval_to_lookback[interval]
            
            train_data = prices[:target_indices[0]].reshape(-1, 1)
            
            self.show_status("Обучение модели...", "blue")
            self.lstm_predictor = LSTMPredictor(train_data, look_back, interval)
            self.lstm_predictor.train(epochs=100)
            
            self.show_status("Построение прогноза...", "blue")
            pred_prices = self.lstm_predictor.predict(len(target_indices))
            
            # Отрисовка прогноза
            self.plot_manager.ax.plot(
                timestamps[target_indices], pred_prices,
                'r--', linewidth=2, label="LSTM Прогноз")
            self.plot_manager.ax.legend()
            self.plot_manager.canvas.draw()
            
            self.show_status("Прогноз построен", "green")

        except ValueError as e:
            self.show_status(str(e), "red")
        except Exception as e:
            self.show_status(f"Ошибка: {str(e)}", "red")
        finally:
            self.plot_manager._setup_zoom_selector()
            self.predict_button.configure(
                text="Прогноз LSTM",
                fg_color="green",
                state="normal"
            )
            # Автоматическое скрытие статуса через 5 сек
            self.after(5000, lambda: self.show_status("", "black"))

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = StockMonitorApp()
    app.mainloop()
'''
'''
#в3\1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from pathlib import Path
import numpy as np
import customtkinter as ctk
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from tensorflow.keras.layers import InputLayer  # Добавлено

from lstm_predictor import LSTMPredictor
from backend.api_client import TinkoffApiClient


class StockMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Stock Monitor Pro")
        self.geometry("1400x900")
        
        # Настройки периодов
        self.period_settings = {
            "1 день": {"interval": "1min", "days": 1},
            "3 дня": {"interval": "5min", "days": 3},
            "1 неделя": {"interval": "15min", "days": 7},
            "1 месяц": {"interval": "day", "days": 30},
            "3 месяца": {"interval": "day", "days": 90},
            "1 год": {"interval": "day", "days": 365}
        }
        
        self.api = TinkoffApiClient()
        self.current_figi = "BBG004730N88"  # SBER по умолчанию
        self.setup_ui()
        
    def setup_ui(self):
        """Инициализация интерфейса"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Панель управления
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        # Поле ввода FIGI
        self.figi_entry = ctk.CTkEntry(
            control_frame, 
            width=200,
            placeholder_text="FIGI инструмента"
        )
        self.figi_entry.pack(side="left", padx=5)
        self.figi_entry.insert(0, self.current_figi)
        
        # Кнопка обновления
        ctk.CTkButton(
            control_frame,
            text="Обновить данные",
            command=self.load_and_plot_data,
            fg_color="#2b5b84"
        ).pack(side="left", padx=5)
        
        # Выбор периода
        self.period_var = ctk.StringVar(value="1 неделя")
        ctk.CTkOptionMenu(
            control_frame,
            values=list(self.period_settings.keys()),
            variable=self.period_var,
            command=self._period_changed
        ).pack(side="left", padx=5)
        
        # Кнопки прогнозирования
        self.predict_btn = ctk.CTkButton(
            control_frame,
            text="Прогноз на участке",
            command=self.activate_range_selection,
            fg_color="#3a7ebf"
        )
        self.predict_btn.pack(side="left", padx=5)
        
        self.future_predict_btn = ctk.CTkButton(
            control_frame,
            text="Прогноз на будущее",
            command=self.activate_future_prediction,
            fg_color="#1f538d"
        )
        self.future_predict_btn.pack(side="left", padx=5)
        
        # Кнопка сброса
        ctk.CTkButton(
            control_frame,
            text="Сбросить вид",
            command=self.reset_plot,
            fg_color="#5d5d5d"
        ).pack(side="left", padx=5)
        
        # Статус бар
        self.status_label = ctk.CTkLabel(
            control_frame, 
            text="Готов к работе",
            text_color="#4CAF50"
        )
        self.status_label.pack(side="right", padx=10)
        
        # Область графика
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Инициализация SpanSelector
        self.span = SpanSelector(
            self.ax,
            self.on_range_selected,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='#3a7ebf'),
            interactive=True
        )
        self.span.set_active(False)
        
        # Привязка событий
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('figure_leave_event', self.on_mouse_leave)
        
    def _period_changed(self, *args):
        """Обработчик изменения периода"""
        self.load_and_plot_data()
        
    def load_and_plot_data(self):
        """Загрузка и отображение данных"""
        self.current_figi = self.figi_entry.get().strip()
        if not self.current_figi:
            self.show_status("Введите FIGI инструмента", "red")
            return
            
        try:
            self.show_status("Загрузка данных...", "blue")
            period = self.period_var.get()
            settings = self.period_settings[period]
            
            candles = self.api.get_candles(
                figi=self.current_figi,
                interval=settings["interval"],
                days=settings["days"]
            )
            
            if not candles:
                self.show_status("Нет данных для отображения", "red")
                return
                
            # Обработка данных
            timestamps = np.array([candle.time for candle in candles])
            prices = np.array([float(candle.close.units) + candle.close.nano/1e9 
                             for candle in candles])
            
            # Сохранение текущих данных
            self.current_data = (timestamps, prices)
            self.current_interval = settings["interval"]
            
            # Отрисовка
            self.ax.clear()
            self.ax.plot(timestamps, prices, 'b-', linewidth=1, label="Цена закрытия")
            
            # Настройка осей
            self._configure_axes(settings["interval"])
            self.ax.set_title(
                f"{self.current_figi} | {period} | {settings['interval']}",
                fontsize=12
            )
            self.ax.grid(True, linestyle='--', alpha=0.5)
            self.ax.legend()
            
            self.canvas.draw()
            self.show_status(f"Загружено {len(prices)} точек", "green")
            
        except Exception as e:
            self.show_status(f"Ошибка: {str(e)}", "red")
            print(f"[ERROR] {e}")
            
    def _configure_axes(self, interval):
        """Настройка отображения осей"""
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(formatter)
        
        if interval == 'day':
            self.ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        elif interval == '1min':
            self.ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            
        self.figure.autofmt_xdate()
        self.ax.yaxis.set_major_formatter('{x:,.2f}')
        
    def activate_range_selection(self):
        """Активация выбора диапазона для прогноза"""
        if not hasattr(self, 'current_data'):
            self.show_status("Сначала загрузите данные", "red")
            return
            
        self.predict_btn.configure(
            text="Выделите участок на графике",
            state="disabled"
        )
        self.show_status("Выделите участок для прогноза...", "blue")
        self.span.set_active(True)
        
    def on_range_selected(self, xmin, xmax):
        """Обработка выбранного диапазона"""
        try:
            self.span.set_active(False)
            self.predict_btn.configure(
                text="Обработка...",
                fg_color="#9e9e9e"
            )
            self.update()
            
            timestamps, prices = self.current_data
            xmin_dt = mdates.num2date(xmin)
            xmax_dt = mdates.num2date(xmax)
            
            mask = (timestamps >= xmin_dt) & (timestamps <= xmax_dt)
            target_indices = np.where(mask)[0]
            
            if len(target_indices) < 20:
                raise ValueError("Выберите участок минимум из 20 точек")
                
            split_idx = target_indices[0]
            
            # Прогнозирование
            self.show_status("Создание модели...", "blue")
            self.lstm_predictor = LSTMPredictor(
                prices[:split_idx].reshape(-1, 1),
                timestamps[:split_idx],
                self.current_interval
            )
            
            self.show_status("Подготовка данных...", "blue")
            X, y = self.lstm_predictor.prepare_data()
            
            self.show_status("Обучение модели...", "blue")
            self.lstm_predictor.train(X, y, epochs=150)
            
            self.show_status("Прогнозирование...", "blue")
            predictions = self.lstm_predictor.predict(len(target_indices))
            
            # Отрисовка результатов
            mae = np.mean(np.abs(predictions.flatten() - prices[target_indices]))
            self.ax.plot(
                timestamps[target_indices],
                predictions,
                'r--',
                linewidth=2,
                label=f"Прогноз (MAE: {mae:.4f})"
            )
            
            # Доверительный интервал
            std = np.std(predictions.flatten() - prices[target_indices])
            self.ax.fill_between(
                timestamps[target_indices],
                predictions.flatten() - std,
                predictions.flatten() + std,
                color='red', alpha=0.1
            )
            
            self.ax.legend()
            self.canvas.draw()
            self.show_status(f"Прогноз готов! MAE: {mae:.4f}", "green")
            
        except Exception as e:
            self.show_status(f"Ошибка: {str(e)}", "red")
            print(f"[ERROR] {e}")
        finally:
            self.predict_btn.configure(
                text="Прогноз на участке",
                fg_color="#3a7ebf",
                state="normal"
            )
            
    def activate_future_prediction(self):
        """Активация прогноза на будущее"""
        if not hasattr(self, 'current_data'):
            self.show_status("Сначала загрузите данные", "red")
            return
            
        dialog = ctk.CTkInputDialog(
            text=f"Введите количество шагов для прогноза (1 шаг = {self.current_interval}):",
            title="Прогноз на будущее"
        )
        
        try:
            steps = int(dialog.get_input())
            if steps <= 0:
                raise ValueError("Число шагов должно быть положительным")
                
            self._make_future_prediction(steps)
        except (ValueError, TypeError):
            self.show_status("Некорректное число шагов", "red")
            
    def _make_future_prediction(self, steps):
        """Прогнозирование будущих значений"""
        try:
            self.future_predict_btn.configure(
                text="Обработка...",
                state="disabled"
            )
            self.show_status("Подготовка...", "blue")
            self.update()
            
            timestamps, prices = self.current_data
            
            # Создание и обучение модели
            self.show_status("Создание модели...", "blue")
            self.lstm_predictor = LSTMPredictor(
                prices.reshape(-1, 1),
                timestamps,
                self.current_interval
            )
            
            X, y = self.lstm_predictor.prepare_data()
            self.lstm_predictor.train(X, y)
            
            # Прогнозирование
            self.show_status("Прогнозирование...", "blue")
            future_predictions = self.lstm_predictor.predict(steps)
            
            # Генерация временных меток
            last_ts = timestamps[-1]
            delta_map = {
                '1min': timedelta(minutes=1),
                '5min': timedelta(minutes=5),
                '15min': timedelta(minutes=15),
                'day': timedelta(days=1)
            }
            future_timestamps = [last_ts + delta_map[self.current_interval] * i 
                               for i in range(1, steps+1)]
            
            # Отрисовка
            self.ax.plot(
                future_timestamps,
                future_predictions,
                'g--',
                linewidth=2,
                label=f"Прогноз на {steps} шагов"
            )
            self.ax.legend()
            self.canvas.draw()
            self.show_status(f"Прогноз на {steps} шагов готов", "green")
            
        except Exception as e:
            self.show_status(f"Ошибка: {str(e)}", "red")
            print(f"[ERROR] {e}")
        finally:
            self.future_predict_btn.configure(
                text="Прогноз на будущее",
                fg_color="#1f538d",
                state="normal"
            )
            
    def reset_plot(self):
        """Сброс графика к исходному виду"""
        if hasattr(self, 'current_data'):
            self.ax.clear()
            timestamps, prices = self.current_data
            self.ax.plot(timestamps, prices, 'b-', linewidth=1)
            self._configure_axes(self.current_interval)
            self.ax.set_title(
                f"{self.current_figi} | {self.period_var.get()} | {self.current_interval}",
                fontsize=12
            )
            self.ax.legend()
            self.canvas.draw()
            self.show_status("График сброшен", "green")
            
    def on_mouse_move(self, event):
        """Отображение цены при наведении"""
        if event.inaxes != self.ax or not hasattr(self, 'current_data'):
            return
            
        timestamps, prices = self.current_data
        xdata = mdates.num2date(event.xdata)
        idx = np.argmin(np.abs(timestamps - xdata))
        
        if 0 <= idx < len(prices):
            price = prices[idx]
            timestamp = timestamps[idx]
            
            if self.current_interval == 'day':
                label = timestamp.strftime('%Y-%m-%d')
            else:
                label = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
            self.status_label.configure(
                text=f"{label} | Цена: {price:.2f}",
                text_color="#2196F3"
            )
            
    def on_mouse_leave(self, event):
        """Сброс статуса при уходе мыши"""
        self.show_status("Готов", "green")
        
    def show_status(self, message, color):
        """Обновление статусной строки"""
        colors = {
            "green": "#4CAF50",
            "red": "#F44336",
            "blue": "#2196F3",
            "yellow": "#FFC107"
        }
        self.status_label.configure(
            text=message,
            text_color=colors.get(color, "#4CAF50")
        )
        self.update()

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = StockMonitorApp()
    app.mainloop()
'''