# core/all_backtest_engine.py

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Tuple
import time
from scipy.signal import find_peaks

from settings.settings import BacktestConfig


class BacktestEngine:
    """
    Оптимизированный движок бэктестинга с vectorized операциями и кешем данных.
    Использует numpy, scipy и кеширование для максимальной производительности.
    """

    # Глобальный кеш данных для всех экземпляров
    _data_cache = {}

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.quiet_mode = False  # Флаг тихого режима

        # Numpy массивы для данных
        self.timestamps = None
        self.ohlc_data = None  # [open, high, low, close]

        # Результаты
        self.pivot_highs: List[Dict] = []
        self.pivot_lows: List[Dict] = []
        self.signals: List[Dict] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # Статистика выполнения
        self.start_time = None
        self.end_time = None
        self.total_pnl = 0.0

    def run_backtest(self, quiet_mode: bool = False) -> bool:
        """Запускает полный цикл оптимизированного бэктестирования"""
        try:
            self.quiet_mode = quiet_mode  # Сохраняем флаг для всех методов
            self.start_time = time.time()

            # 1. Загружаем данные из кеша или создаем кеш
            if not self._load_data_cached():
                return False

            # 2. Супер-векторизованный расчет пивотов с scipy
            pivot_high_indices, pivot_low_indices = self._calculate_pivots_scipy()

            # 3. Конвертируем пивоты в список для совместимости
            self._convert_pivots_to_legacy_format(pivot_high_indices, pivot_low_indices)

            # 4. Оптимизированная симуляция стратегии
            self._simulate_strategy_optimized()

            # 5. Векторизованный расчет результатов
            self._calculate_results_vectorized()

            self.end_time = time.time()

            # Выводим результаты только если не тихий режим
            if not quiet_mode:
                self._print_results()

            return True

        except Exception as e:
            if not quiet_mode:
                print(f"Ошибка бэктестирования: {e}")
                import traceback
                traceback.print_exc()
            return False

    def _load_data_cached(self) -> bool:
        """Загружает данные из глобального кеша или создает кеш"""
        try:
            csv_path = self.config.data.csv_path

            # Проверяем наличие файла
            if not os.path.exists(csv_path):
                print(f"Ошибка: Файл не найден: {csv_path}")
                return False

            # Проверяем кеш
            if csv_path not in BacktestEngine._data_cache:
                if not self.quiet_mode:
                    print(f"Загрузка данных в кеш: {csv_path}")

                # Загружаем данные один раз для всех процессов
                data = pd.read_csv(csv_path)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp').reset_index(drop=True)

                # Сохраняем в кеш как numpy массивы
                BacktestEngine._data_cache[csv_path] = {
                    'timestamps': data['timestamp'].values,
                    'ohlc_data': np.column_stack([
                        data['open'].values,
                        data['high'].values,
                        data['low'].values,
                        data['close'].values
                    ]).astype(np.float64),
                    'data_length': len(data)
                }

                if not self.quiet_mode:
                    print(f"Кеш создан: {len(data)} баров")
            else:
                if not self.quiet_mode:
                    print("Данные загружены из кеша")

            # Получаем данные из кеша
            cached_data = BacktestEngine._data_cache[csv_path]
            self.timestamps = cached_data['timestamps']
            self.ohlc_data = cached_data['ohlc_data']

            if not self.quiet_mode:
                print(f"Используется {len(self.timestamps)} баров")
                print(f"Период: {self.timestamps[0]} - {self.timestamps[-1]}")

            return True

        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return False

    @classmethod
    def clear_data_cache(cls):
        """Очищает кеш данных (полезно для тестов или смены файлов)"""
        cls._data_cache.clear()

    def _calculate_pivots_scipy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Супер-векторизованный расчет пивотов с использованием scipy.signal
        Значительно быстрее чем ручная реализация
        """
        left_bars = self.config.pivot.left_bars
        right_bars = self.config.pivot.right_bars

        # Минимальное расстояние между пивотами
        min_distance = left_bars + right_bars + 1

        high_prices = self.ohlc_data[:, 1]  # High prices
        low_prices = self.ohlc_data[:, 2]  # Low prices

        # Находим пики (максимумы) для pivot highs
        pivot_high_indices, _ = find_peaks(
            high_prices,
            distance=min_distance,
            width=None  # Не ограничиваем ширину пика
        )

        # Находим пики перевернутого массива для pivot lows (минимумы)
        pivot_low_indices, _ = find_peaks(
            -low_prices,  # Инвертируем для поиска минимумов
            distance=min_distance,
            width=None
        )

        # Фильтруем пивоты с учетом left_bars и right_bars границ
        valid_high_mask = (pivot_high_indices >= left_bars) & (pivot_high_indices < len(high_prices) - right_bars)
        pivot_high_indices = pivot_high_indices[valid_high_mask]

        valid_low_mask = (pivot_low_indices >= left_bars) & (pivot_low_indices < len(low_prices) - right_bars)
        pivot_low_indices = pivot_low_indices[valid_low_mask]

        # Дополнительная проверка на соответствие критериям пивота
        pivot_high_indices = BacktestEngine._validate_pivots_vectorized(
            pivot_high_indices, high_prices, left_bars, right_bars, find_maxima=True
        )

        pivot_low_indices = BacktestEngine._validate_pivots_vectorized(
            pivot_low_indices, low_prices, left_bars, right_bars, find_maxima=False
        )

        if not self.quiet_mode:
            print(f"Найдено pivot highs: {len(pivot_high_indices)}")
            print(f"Найдено pivot lows: {len(pivot_low_indices)}")

        return pivot_high_indices, pivot_low_indices

    @staticmethod
    def _validate_pivots_vectorized(candidate_indices: np.ndarray, prices: np.ndarray,
                                    left_bars: int, right_bars: int, find_maxima: bool) -> np.ndarray:
        """
        Векторизованная проверка пивотов на соответствие критериям
        """
        if len(candidate_indices) == 0:
            return candidate_indices

        valid_pivots = []

        for idx in candidate_indices:
            # Получаем окно вокруг кандидата
            start_idx = max(0, idx - left_bars)
            end_idx = min(len(prices), idx + right_bars + 1)
            window = prices[start_idx:end_idx]
            candidate_price = prices[idx]

            if find_maxima:
                # Для pivot high: должен быть строго больше всех соседей
                # или равен максимуму и больше хотя бы одного соседа
                neighbors = np.concatenate([window[:idx - start_idx], window[idx - start_idx + 1:]])
                if len(neighbors) > 0 and candidate_price >= window.max() and candidate_price > neighbors.max():
                    valid_pivots.append(idx)
            else:
                # Для pivot low: должен быть строго меньше всех соседей
                # или равен минимуму и меньше хотя бы одного соседа
                neighbors = np.concatenate([window[:idx - start_idx], window[idx - start_idx + 1:]])
                if len(neighbors) > 0 and candidate_price <= window.min() and candidate_price < neighbors.min():
                    valid_pivots.append(idx)

        return np.array(valid_pivots, dtype=int)

    def _convert_pivots_to_legacy_format(self, pivot_high_indices: np.ndarray,
                                         pivot_low_indices: np.ndarray) -> None:
        """Конвертирует numpy индексы в legacy формат для совместимости"""
        right_bars = self.config.pivot.right_bars

        # Pivot highs
        self.pivot_highs = []
        for idx in pivot_high_indices:
            if idx + right_bars < len(self.timestamps):
                confirmed_idx = idx + right_bars
                self.pivot_highs.append({
                    'timestamp': self.timestamps[idx],
                    'price': self.ohlc_data[idx, 1],  # High price
                    'confirmed_at': self.timestamps[confirmed_idx],
                    'bar_index': idx
                })

        # Pivot lows
        self.pivot_lows = []
        for idx in pivot_low_indices:
            if idx + right_bars < len(self.timestamps):
                confirmed_idx = idx + right_bars
                self.pivot_lows.append({
                    'timestamp': self.timestamps[idx],
                    'price': self.ohlc_data[idx, 2],  # Low price
                    'confirmed_at': self.timestamps[confirmed_idx],
                    'bar_index': idx
                })

    def _simulate_strategy_optimized(self) -> None:
        """
        Оптимизированная симуляция стратегии с использованием numpy
        """
        self.signals = []
        self.trades = []

        # Создаем массивы для быстрого поиска пивотов по времени подтверждения
        pivot_confirmations = {}

        for pivot in self.pivot_highs:
            confirm_time = pivot['confirmed_at']
            if confirm_time not in pivot_confirmations:
                pivot_confirmations[confirm_time] = []
            pivot_confirmations[confirm_time].append({
                'type': 'HIGH',
                'price': pivot['price'],
                'pivot_time': pivot['timestamp']
            })

        for pivot in self.pivot_lows:
            confirm_time = pivot['confirmed_at']
            if confirm_time not in pivot_confirmations:
                pivot_confirmations[confirm_time] = []
            pivot_confirmations[confirm_time].append({
                'type': 'LOW',
                'price': pivot['price'],
                'pivot_time': pivot['timestamp']
            })

        # Состояние стратегии
        hprice = 0.0
        lprice = 0.0
        le = False
        se = False
        current_position = None
        current_entry = None

        # Векторизованный проход по барам
        for i, timestamp in enumerate(self.timestamps):
            current_high = self.ohlc_data[i, 1]
            current_low = self.ohlc_data[i, 2]

            # 1. Проверяем подтверждение пивотов
            if timestamp in pivot_confirmations:
                for pivot_event in pivot_confirmations[timestamp]:
                    if pivot_event['type'] == 'HIGH':
                        hprice = pivot_event['price']
                        le = True
                    else:  # LOW
                        lprice = pivot_event['price']
                        se = True

            # 2. Проверяем сигналы
            if le and 0 < hprice < current_high and current_position != "LONG":
                entry_price = hprice + self.config.trading.min_tick

                # Закрываем SHORT если есть
                if current_position == "SHORT":
                    self._close_position(current_entry, timestamp, entry_price, "LONG_SIGNAL")

                # Открываем LONG
                self._open_position("LONG", timestamp, entry_price)
                current_position = "LONG"
                current_entry = {'time': timestamp, 'price': entry_price}
                le = False

            elif se and lprice > 0 and current_low < lprice and current_position != "SHORT":
                entry_price = lprice - self.config.trading.min_tick

                # Закрываем LONG если есть
                if current_position == "LONG":
                    self._close_position(current_entry, timestamp, entry_price, "SHORT_SIGNAL")

                # Открываем SHORT
                self._open_position("SHORT", timestamp, entry_price)
                current_position = "SHORT"
                current_entry = {'time': timestamp, 'price': entry_price}
                se = False

            # 3. Деактивируем условия при пробое
            if le and current_high > hprice and current_position == "LONG":
                le = False
            if se and current_low < lprice and current_position == "SHORT":
                se = False

        if not self.quiet_mode:
            print(f"Сгенерировано сигналов: {len(self.signals)}")
            print(f"Совершено сделок: {len(self.trades)}")

    def _open_position(self, direction: str, timestamp, entry_price: float) -> None:
        """Открывает позицию и записывает сигнал"""
        self.signals.append({
            'timestamp': timestamp,
            'type': direction,
            'price': entry_price
        })

    def _close_position(self, entry_info: Dict, exit_time, exit_price: float,
                        exit_reason: str) -> None:
        """Закрывает позицию и записывает сделку"""
        position_volume = self.config.trading.position_size * self.config.trading.leverage

        # Определяем направление по цене входа vs выхода
        if exit_reason == "LONG_SIGNAL":
            direction = "SHORT"
            entry_type = "SHORT"
        else:  # SHORT_SIGNAL
            direction = "LONG"
            entry_type = "LONG"

        trade = {
            'entry_time': entry_info['time'],
            'entry_price': entry_info['price'],
            'entry_type': entry_type,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_type': exit_reason,
            'quantity': position_volume,
            'symbol': self.config.data.symbol,
            'direction': direction
        }

        self.trades.append(trade)

    def _calculate_results_vectorized(self) -> None:
        """Векторизованный расчет результатов"""
        if not self.trades:
            self.total_pnl = 0.0
            self._create_equity_curve_empty()
            return

        # Векторизованный расчет PnL для всех сделок
        trades_array = np.array([
            [trade['entry_price'], trade['exit_price'],
             1.0 if trade['entry_type'] == 'LONG' else -1.0]
            for trade in self.trades
        ])

        position_volume = self.config.trading.position_size * self.config.trading.leverage

        # Векторизованный расчет изменения цен и PnL
        entry_prices = trades_array[:, 0]
        exit_prices = trades_array[:, 1]
        directions = trades_array[:, 2]

        # Расчет процентного изменения с учетом направления
        price_changes = (exit_prices - entry_prices) / entry_prices * directions

        # PnL до комиссий
        pnl_before_commission = position_volume * price_changes

        # Комиссии
        total_commission = position_volume * self.config.trading.taker_commission * 2

        # Итоговый PnL
        final_pnls = pnl_before_commission - total_commission

        # Записываем результаты в сделки
        for i, trade in enumerate(self.trades):
            trade['pnl'] = final_pnls[i]
            trade['commission'] = total_commission

        self.total_pnl = final_pnls.sum()
        self._create_equity_curve_optimized()

    def _create_equity_curve_optimized(self) -> None:
        """Создает оптимизированную кривую капитала"""
        self.equity_curve = []
        current_capital = self.config.trading.initial_capital

        # Начальная точка
        self.equity_curve.append({
            'timestamp': self.timestamps[0],
            'equity': current_capital,
            'price': self.ohlc_data[0, 3],  # Close price
            'total_pnl': 0.0,
            'unrealized_pnl': 0.0
        })

        # Создаем индекс времени для быстрого поиска
        timestamp_to_idx = {ts: i for i, ts in enumerate(self.timestamps)}

        # Добавляем точки после каждой сделки
        for trade in self.trades:
            current_capital += trade['pnl']

            # Быстрый поиск цены закрытия
            bar_idx = timestamp_to_idx.get(trade['exit_time'])
            close_price = self.ohlc_data[bar_idx, 3] if bar_idx is not None else trade['exit_price']

            self.equity_curve.append({
                'timestamp': trade['exit_time'],
                'equity': current_capital,
                'price': close_price,
                'total_pnl': current_capital - self.config.trading.initial_capital,
                'unrealized_pnl': 0.0
            })

        # Финальная точка
        if self.equity_curve[-1]['timestamp'] != self.timestamps[-1]:
            self.equity_curve.append({
                'timestamp': self.timestamps[-1],
                'equity': current_capital,
                'price': self.ohlc_data[-1, 3],  # Final close
                'total_pnl': current_capital - self.config.trading.initial_capital,
                'unrealized_pnl': 0.0
            })

    def _create_equity_curve_empty(self) -> None:
        """Создает пустую equity curve если нет сделок"""
        if len(self.timestamps) == 0:
            return

        self.equity_curve = [
            {
                'timestamp': self.timestamps[0],
                'equity': self.config.trading.initial_capital,
                'price': self.ohlc_data[0, 3],
                'total_pnl': 0.0,
                'unrealized_pnl': 0.0
            },
            {
                'timestamp': self.timestamps[-1],
                'equity': self.config.trading.initial_capital,
                'price': self.ohlc_data[-1, 3],
                'total_pnl': 0.0,
                'unrealized_pnl': 0.0
            }
        ]

    def _print_results(self) -> None:
        """Выводит результаты бэктеста"""
        execution_time = self.end_time - self.start_time
        print("-" * 60)
        print("РЕЗУЛЬТАТЫ ОПТИМИЗИРОВАННОГО БЭКТЕСТА")
        print("-" * 60)
        print(f"Время выполнения: {execution_time:.2f} секунд")
        print(f"Обработано баров: {len(self.timestamps):,}")
        print(f"Скорость: {len(self.timestamps) / execution_time:,.0f} баров/сек")
        print(f"Начальный капитал: ${self.config.trading.initial_capital:,.2f}")
        print(f"Общая прибыль: ${self.total_pnl:,.2f}")
        final_capital = self.config.trading.initial_capital + self.total_pnl
        print(f"Итоговый капитал: ${final_capital:,.2f}")
        print(f"Доходность: {(final_capital / self.config.trading.initial_capital - 1) * 100:.2f}%")

    def get_results(self) -> Dict[str, Any]:
        """Возвращает результаты для аналитики - совместимый API"""
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'signals': self.signals,
            'pivot_highs': self.pivot_highs,
            'pivot_lows': self.pivot_lows,
            'settings': self.config.to_dict(),
            'total_pnl': self.total_pnl,
            'execution_stats': {
                'execution_time': self.end_time - self.start_time if self.end_time else 0,
                'total_bars': len(self.timestamps) if self.timestamps is not None else 0,
                'total_signals': len(self.signals),
                'total_trades': len(self.trades)
            }
        }