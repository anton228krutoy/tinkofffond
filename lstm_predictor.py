
#в1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Полностью отключаем логи TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings

# Отключаем все предупреждения
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class LSTMPredictor:
    def __init__(self, data, look_back=60):
        self.data = data
        self.look_back = look_back
        self.scaler = MinMaxScaler()
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.look_back, 1)))  # Явный слой ввода
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self):
        scaled_data = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def train(self, epochs=10, batch_size=32):
        X, y = self.prepare_data()
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        return history

    def predict(self, prediction_days=30):
        inputs = self.data[-self.look_back:]
        inputs = self.scaler.transform(inputs)
        predictions = []

        for _ in range(prediction_days):
            x_input = inputs[-self.look_back:].reshape(1, self.look_back, 1)
            pred = self.model.predict(x_input, verbose=0)
            inputs = np.append(inputs, pred)
            predictions.append(pred[0][0])
            
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions

if __name__ == "__main__":
    # Тестовые данные
    days = 365
    test_data = np.sin(np.linspace(0, 10, days)) * 50 + 100
    test_data = test_data.reshape(-1, 1)

    # Инициализация и обучение
    predictor = LSTMPredictor(test_data)
    print("Обучение модели...")
    predictor.train(epochs=5)

    # Прогноз и визуализация
    predictions = predictor.predict(30)
    plt.figure(figsize=(10, 5))
    plt.plot(test_data, label='Исторические данные')
    plt.plot(range(days, days+30), predictions, 'r--', label='Прогноз')
    plt.legend()
    plt.show()

'''
#в2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import warnings
from typing import Tuple

warnings.filterwarnings("ignore")

class LSTMPredictor:
    def __init__(self, data: np.ndarray, look_back: int = 60):
        self.data = data
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_improved_model()
        self.train_history = None

    def _build_improved_model(self) -> Sequential:
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), 
                         input_shape=(self.look_back, 1)),
            Dropout(0.3),
            Bidirectional(LSTM(100)),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        scaled_data = self.scaler.fit_transform(self.data)
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def train(self, epochs: int = 50, batch_size: int = 32, 
             patience: int = 10) -> dict:
        X, y = self.prepare_data()
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        callbacks = [
            EarlyStopping(monitor='loss', patience=patience, 
                         restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.1, 
                            patience=patience//2)
        ]
        
        self.train_history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self.train_history.history

    def predict(self, prediction_days: int) -> np.ndarray:
        if len(self.data) < self.look_back:
            raise ValueError(f"Need at least {self.look_back} points")
            
        last_sequence = self.data[-self.look_back:]
        scaled_seq = self.scaler.transform(last_sequence)
        
        predictions = []
        current_batch = scaled_seq.reshape(1, self.look_back, 1)
        
        for _ in range(prediction_days):
            next_pred = self.model.predict(current_batch, verbose=0)[0, 0]
            predictions.append(next_pred)
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0, -1, 0] = next_pred
            
        return self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)).flatten()
'''

'''
#в3
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

class LSTMPredictor:
    def __init__(self, data: np.ndarray, look_back: int, interval: str):
        self.data = data
        self.look_back = look_back
        self.interval = interval
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True)), 
            Dropout(0.3),
            Bidirectional(LSTM(100)),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def prepare_data(self):
        scaled_data = self.scaler.fit_transform(self.data)
        X, y = [], []
        
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def train(self, epochs=50, batch_size=32):
        X, y = self.prepare_data()
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='loss', patience=10)],
            verbose=1
        )
        return history
    
    def predict(self, steps: int):
        last_sequence = self.data[-self.look_back:]
        scaled_seq = self.scaler.transform(last_sequence)
        
        predictions = []
        current_batch = scaled_seq.reshape(1, self.look_back, 1)
        
        for _ in range(steps):
            pred = self.model.predict(current_batch, verbose=0)[0, 0]
            predictions.append(pred)
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0, -1, 0] = pred
        
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
'''

'''
#в3\1
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class LSTMPredictor:
    def __init__(self, data: np.ndarray, timestamps: np.ndarray, interval: str):
        self.data = data
        self.timestamps = timestamps
        self.interval = interval
        self.scaler = MinMaxScaler()
        self.look_back = self._calculate_lookback()
        self.model = self._build_model()

    def _calculate_lookback(self) -> int:
        """Безопасный расчет длины окна"""
        min_points = max(20, len(self.data) // 5)  # Минимум 20 точек
        return min(60, min_points)  # Ограничение сверху

    def _build_model(self):
        model = Sequential([
            InputLayer(input_shape=(self.look_back, 1)),
            LSTM(64, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self):
        """Упрощенная подготовка данных"""
        scaled_data = self.scaler.fit_transform(self.data)
        X, y = [], []
        
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X).reshape(-1, self.look_back, 1), np.array(y)

    def train(self, X, y, epochs=10):
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def predict(self, steps: int):
        last_window = self.scaler.transform(self.data[-self.look_back:])
        predictions = []
        
        for _ in range(steps):
            pred = self.model.predict(last_window.reshape(1, self.look_back, 1), verbose=0)
            predictions.append(pred[0,0])
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred[0,0]
            
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
'''