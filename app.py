import sys
from pathlib import Path
import tkinter as tk
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.widgets import SpanSelector

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

from backend.api_client import TinkoffApiClient
from backend.data_processor import DataProcessor

class StockMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Stock Monitor with Zoom")
        self.geometry("1400x900")
        self._setup_ui()
        self.api = TinkoffApiClient()
        self.processor = DataProcessor()
        self.original_xlim = None
        self.original_ylim = None

    def _setup_ui(self):
        """Настройка интерфейса с работающим зумом"""
        # Главный контейнер
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Панель управления
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Поле для FIGI
        self.figi_entry = ctk.CTkEntry(
            control_frame,
            placeholder_text="Введите FIGI",
            width=300
        )
        self.figi_entry.pack(side=tk.LEFT, padx=5)
        self.figi_entry.insert(0, "BBG004730N88")
        
        # Кнопка обновления
        self.update_btn = ctk.CTkButton(
            control_frame,
            text="Обновить график",
            command=self.update_chart
        )
        self.update_btn.pack(side=tk.LEFT, padx=5)
        
        # Кнопка сброса зума
        self.zoom_out_btn = ctk.CTkButton(
            control_frame,
            text="Сбросить zoom",
            command=self.reset_zoom,
            fg_color="gray"
        )
        self.zoom_out_btn.pack(side=tk.LEFT, padx=5)
        
        # Область графика
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Инициализация SpanSelector для зума
        self.span = SpanSelector(
            self.ax,
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='yellow'),
            interactive=True
        )

    def on_select(self, xmin, xmax):
        """Обработчик выделения области для зума"""
        if self.original_xlim is None:
            self.original_xlim = self.ax.get_xlim()
            self.original_ylim = self.ax.get_ylim()
        
        self.ax.set_xlim(xmin, xmax)
        self.canvas.draw_idle()

    def reset_zoom(self):
        """Сброс зума к исходному виду"""
        if self.original_xlim:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.canvas.draw_idle()
            self.original_xlim = None

    def update_chart(self):
        """Обновление графика с минутными данными"""
        figi = self.figi_entry.get().strip()
        if not figi:
            return
            
        try:
            # Получаем минутные данные
            candles = self.api.get_candles(figi, "1min", 1)
            prices, timestamps = self.processor.process_candles(candles)
            
            # Очищаем и перерисовываем график
            self.ax.clear()
            line, = self.ax.plot(timestamps, prices, 'b-', linewidth=1, label=f"{figi} (1min)")
            
            # Настраиваем отображение
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.ax.set_xlabel('Время (МСК)')
            self.ax.set_ylabel('Цена, руб')
            self.ax.set_title(f'Минутный график {figi} (выделите область для зума)')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            
            # Сохраняем исходные границы
            self.original_xlim = self.ax.get_xlim()
            self.original_ylim = self.ax.get_ylim()
            
            # Обновляем SpanSelector
            self.span = SpanSelector(
                self.ax,
                self.on_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.5, facecolor='yellow'),
                interactive=True
            )
            
            self.fig.autofmt_xdate()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = StockMonitorApp()
    app.mainloop()