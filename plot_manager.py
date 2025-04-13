'''
#в2
import numpy as np
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from typing import Optional, Tuple

class PlotManager:
    def __init__(self, master):
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.current_data = None
        self.original_limits = None
        
        # Инициализация методов
        self.setup_annotation()
        self.setup_zoom_selector()

    def setup_annotation(self):
        """Настройка аннотации для отображения цены"""
        self.price_annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"))
        self.price_annotation.set_visible(False)

    def setup_zoom_selector(self):
        """Настройка SpanSelector для зума"""
        self.span_zoom = SpanSelector(
            self.ax,
            self.on_zoom_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='yellow'),
            interactive=True
        )

    def setup_range_selection(self, callback):
        """Временный режим выбора диапазона для прогноза"""
        self.span_zoom.set_active(False)  # Отключаем зум
        
        self.span_predict = SpanSelector(
            self.ax,
            callback,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red')
        )

    def reset_zoom_selector(self):
        """Возвращаем обычный режим зума"""
        if hasattr(self, 'span_predict'):
            self.span_predict.set_active(False)
        self.span_zoom.set_active(True)

    def plot_data(self, timestamps, prices, title, interval):
        """Отрисовка новых данных"""
        self.ax.clear()
        self.ax.plot(timestamps, prices, 'b-', linewidth=1)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        
        # Настройка осей
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(formatter)
        self.fig.autofmt_xdate()
        self.ax.yaxis.set_major_formatter('{x:,.2f}')
        
        self.current_data = (timestamps, prices)
        self.original_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self.canvas.draw()

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
            y_min = np.min(visible_data)
            y_max = np.max(visible_data)
            y_padding = (y_max - y_min) * 0.05
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            self.canvas.draw_idle()

    def reset_zoom(self):
        """Сброс зума к исходному виду"""
        if self.original_limits:
            self.ax.set_xlim(self.original_limits[0])
            self.ax.set_ylim(self.original_limits[1])
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Отображение цены при наведении"""
        if event.inaxes != self.ax or self.current_data is None:
            self.price_annotation.set_visible(False)
            self.canvas.draw_idle()
            return
            
        timestamps, prices = self.current_data
        xdata = mdates.num2date(event.xdata)
        idx = np.argmin(np.abs(timestamps - xdata))
        
        if 0 <= idx < len(prices):
            price = prices[idx]
            timestamp = timestamps[idx]
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            self.price_annotation.xy = (timestamp, price)
            self.price_annotation.set_text(f"{time_str}\nЦена: {price:.2f}")
            self.price_annotation.set_visible(True)
            self.canvas.draw_idle()

    def on_mouse_leave(self, event):
        """Скрытие аннотации при уходе мыши"""
        self.price_annotation.set_visible(False)
        self.canvas.draw_idle()
'''

#в3
import numpy as np
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from typing import Optional, Tuple

class PlotManager:
    def __init__(self, master):
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.current_data = None
        self.original_limits = None
        self.span = None
        self._setup_annotation()
        self._setup_zoom_selector()

    def _setup_annotation(self):
        self.price_annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"))
        self.price_annotation.set_visible(False)

    def _setup_zoom_selector(self):
        if self.span:
            self.span.set_active(False)
        self.span = SpanSelector(
            self.ax,
            self.on_zoom_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='yellow'),
            interactive=True
        )

    def setup_range_selection(self, callback):
        if self.span:
            self.span.set_active(False)
        self.span = SpanSelector(
            self.ax,
            callback,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='red'),
            interactive=True
        )
        self.canvas.draw_idle()

    def plot_data(self, timestamps, prices, title, interval):
        self.ax.clear()
        self.ax.plot(timestamps, prices, 'b-', linewidth=1)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self._configure_axes(interval)
        self.current_data = (timestamps, prices)
        self.original_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self.canvas.draw()

    def _configure_axes(self, interval):
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(formatter)
        self.fig.autofmt_xdate()
        self.ax.yaxis.set_major_formatter('{x:,.2f}')

    def on_zoom_select(self, xmin, xmax):
        if self.current_data is None:
            return
            
        timestamps, prices = self.current_data
        xmin_dt = mdates.num2date(xmin)
        xmax_dt = mdates.num2date(xmax)
        
        mask = (timestamps >= xmin_dt) & (timestamps <= xmax_dt)
        visible_data = prices[mask]
        
        if len(visible_data) > 0:
            self.ax.set_xlim(xmin, xmax)
            y_min = np.min(visible_data)
            y_max = np.max(visible_data)
            y_padding = (y_max - y_min) * 0.05
            self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            self.canvas.draw_idle()

    def reset_zoom(self):
        if self.original_limits:
            self.ax.set_xlim(self.original_limits[0])
            self.ax.set_ylim(self.original_limits[1])
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or self.current_data is None:
            self.price_annotation.set_visible(False)
            self.canvas.draw_idle()
            return
            
        timestamps, prices = self.current_data
        xdata = mdates.num2date(event.xdata)
        idx = np.argmin(np.abs(timestamps - xdata))
        
        if 0 <= idx < len(prices):
            price = prices[idx]
            timestamp = timestamps[idx]
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            self.price_annotation.xy = (timestamp, price)
            self.price_annotation.set_text(f"{time_str}\nЦена: {price:.2f}")
            self.price_annotation.set_visible(True)
            self.canvas.draw_idle()

    def on_mouse_leave(self, event):
        self.price_annotation.set_visible(False)
        self.canvas.draw_idle()