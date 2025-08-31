# core/all_backtest_engine.py

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Tuple
import time
from scipy.signal import find_peaks

from settings.settings import BacktestConfig


class AllBacktestEngine:
    """
    Движок для массовой оптимизации (all backtest).
    Использует кеширование данных, scipy и векторизацию для максимальной скорости.
    """

    # Глобальный кеш данных для всех экземпляров
    _data_cache = {}

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.quiet_mode = False

        # Numpy массивы для данных
        self.timestamps = None
        self.ohlc_data = None

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
        """Запускает all backtest"""
        try:
            self.quiet_mode = quiet_mode
            self.start_time = time.time()

            # 1. Загружаем данные из кеша
            if not self._load_data_cached():
                return False

            # 2. Быстрый расчет пивотов с scipy
            pivot_high_indices, pivot_low_indices = self._calculate_pivots_scipy()

            # 3. Конвертируем пивоты в legacy формат
            self._convert_pivots_to_legacy_format(pivot_high_indices, pivot_low_indices)

            # 4. Оптимизированная симуляция стратегии
            self._simulate_strategy_optimized()

            # 5. Векторизованный расчет результатов
            self._calculate_results_vectorized()

            self.end_time = time.time()

            if not quiet_mode:
                self._print_results()

            return True

        except Exception as e:
            if not quiet_mode:
                print(f"Ошибка all бэктестирования: {e}")
                import traceback
                traceback.print_exc()
            return False

    def _load_data_cached(self) -> bool:
        """Загружает данные из глобального кеша"""
        try:
            csv_path = self.config.data.csv_path

            if not os.path.exists(csv_path):
                print(f"Ошибка: Файл не найден: {csv_path}")
                return False

            # Проверяем кеш
            if csv_path not in AllBacktestEngine._data_cache:
                if not self.quiet_mode:
                    print(f"Загрузка данных в кеш: {csv_path}")

                # Загружаем данные один раз
                data = pd.read_csv(csv_path)
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.sort_values('timestamp').reset_index(drop=True)

                # Сохраняем в кеш как numpy массивы
                AllBacktestEngine._data_cache[csv_path] = {
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

            # Получаем данные из кеша
            cached_data = AllBacktestEngine._data_cache[csv_path]
            self.timestamps = cached_data['timestamps']
            self.ohlc_data = cached_data['ohlc_data']

            if not self.quiet_mode:
                print(f"Используется {len(self.timestamps)} баров из кеша")

            return True

        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return False

    @classmethod
    def clear_data_cache(cls):
        """Очищает кеш данных"""
        cls._data_cache.clear()

    def _calculate_pivots_scipy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Быстрый расчет пивотов с scipy"""
        left_bars = self.config.pivot.left_bars
        right_bars = self.config.pivot.right_bars
        min_distance = left_bars + right_bars + 1

        high_prices = self.ohlc_data[:, 1]
        low_prices = self.ohlc_data[:, 2]

        # Находим пики с scipy
        pivot_high_indices, _ = find_peaks(high_prices, distance=min_distance)
        pivot_low_indices, _ = find_peaks(-low_prices, distance=min_distance)

        # Фильтруем по границам
        valid_high_mask = (pivot_high_indices >= left_bars) & (pivot_high_indices < len(high_prices) - right_bars)
        pivot_high_indices = pivot_high_indices[valid_high_mask]

        valid_low_mask = (pivot_low_indices >= left_bars) & (pivot_low_indices < len(low_prices) - right_bars)
        pivot_low_indices = pivot_low_indices[valid_low_mask]

        # Дополнительная проверка критериев
        pivot_high_indices = AllBacktestEngine._validate_pivots_vectorized(
            pivot_high_indices, high_prices, left_bars, right_bars, find_maxima=True
        )

        pivot_low_indices = AllBacktestEngine._validate_pivots_vectorized(
            pivot_low_indices, low_prices, left_bars, right_bars, find_maxima=False
        )

        if not self.quiet_mode:
            print(f"Найдено pivot highs: {len(pivot_high_indices)}")
            print(f"Найдено pivot lows: {len(pivot_low_indices)}")

        return pivot_high_indices, pivot_low_indices

    @staticmethod
    def _validate_pivots_vectorized(candidate_indices: np.ndarray, prices: np.ndarray,
                                    left_bars: int, right_bars: int, find_maxima: bool) -> np.ndarray:
        """Векторизованная проверка пивотов"""
        if len(candidate_indices) == 0:
            return candidate_indices

        valid_pivots = []

        for idx in candidate_indices:
            start_idx = max(0, idx - left_bars)
            end_idx = min(len(prices), idx + right_bars + 1)
            window = prices[start_idx:end_idx]
            candidate_price = prices[idx]

            if find_maxima:
                neighbors = np.concatenate([window[:idx - start_idx], window[idx - start_idx + 1:]])
                if len(neighbors) > 0 and candidate_price >= window.max() and candidate_price > neighbors.max():
                    valid_pivots.append(idx)
            else:
                neighbors = np.concatenate([window[:idx - start_idx], window[idx - start_idx + 1:]])
                if len(neighbors) > 0 and candidate_price <= window.min() and candidate_price < neighbors.min():
                    valid_pivots.append(idx)

        return np.array(valid_pivots, dtype=int)

    def _convert_pivots_to_legacy_format(self, pivot_high_indices: np.ndarray,
                                         pivot_low_indices: np.ndarray) -> None:
        """Конвертирует numpy индексы в legacy формат"""
        right_bars = self.config.pivot.right_bars

        # Pivot highs
        self.pivot_highs = []
        for idx in pivot_high_indices:
            if idx + right_bars < len(self.timestamps):
                confirmed_idx = idx + right_bars
                self.pivot_highs.append({
                    'timestamp': self.timestamps[idx],
                    'price': self.ohlc_data[idx, 1],
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
                    'price': self.ohlc_data[idx, 2],
                    'confirmed_at': self.timestamps[confirmed_idx],
                    'bar_index': idx
                })

    def _simulate_strategy_optimized(self) -> None:
        """Оптимизированная симуляция стратегии"""
        self.signals = []
        self.trades = []

        # Быстрый поиск пивотов
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

        # Векторизованный проход
        for i, timestamp in enumerate(self.timestamps):
            current_high = self.ohlc_data[i, 1]
            current_low = self.ohlc_data[i, 2]

            # Проверяем подтверждение пивотов
            if timestamp in pivot_confirmations:
                for pivot_event in pivot_confirmations[timestamp]:
                    if pivot_event['type'] == 'HIGH':
                        hprice = pivot_event['price']
                        le = True
                    else:
                        lprice = pivot_event['price']
                        se = True

            # Проверяем сигналы
            if le and 0 < hprice < current_high and current_position != "LONG":
                entry_price = hprice + self.config.trading.min_tick

                if current_position == "SHORT":
                    self._close_position(current_entry, timestamp, entry_price, "LONG_SIGNAL")

                self._open_position("LONG", timestamp, entry_price)
                current_position = "LONG"
                current_entry = {'time': timestamp, 'price': entry_price}
                le = False

            elif se and lprice > 0 and current_low < lprice and current_position != "SHORT":
                entry_price = lprice - self.config.trading.min_tick

                if current_position == "LONG":
                    self._close_position(current_entry, timestamp, entry_price, "SHORT_SIGNAL")

                self._open_position("SHORT", timestamp, entry_price)
                current_position = "SHORT"
                current_entry = {'time': timestamp, 'price': entry_price}
                se = False

            # Деактивация
            if le and current_high > hprice and current_position == "LONG":
                le = False
            if se and current_low < lprice and current_position == "SHORT":
                se = False

        if not self.quiet_mode:
            print(f"Сгенерировано сигналов: {len(self.signals)}")
            print(f"Совершено сделок: {len(self.trades)}")

    def _open_position(self, direction: str, timestamp, entry_price: float) -> None:
        """Открывает позицию"""
        self.signals.append({
            'timestamp': timestamp,
            'type': direction,
            'price': entry_price
        })

    def _close_position(self, entry_info: Dict, exit_time, exit_price: float,
                        exit_reason: str) -> None:
        """Закрывает позицию"""
        position_volume = self.config.trading.position_size * self.config.trading.leverage

        if exit_reason == "LONG_SIGNAL":
            direction = "SHORT"
            entry_type = "SHORT"
        else:
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

        # Векторизованный расчет
        trades_array = np.array([
            [trade['entry_price'], trade['exit_price'],
             1.0 if trade['entry_type'] == 'LONG' else -1.0]
            for trade in self.trades
        ])

        position_volume = self.config.trading.position_size * self.config.trading.leverage

        entry_prices = trades_array[:, 0]
        exit_prices = trades_array[:, 1]
        directions = trades_array[:, 2]

        price_changes = (exit_prices - entry_prices) / entry_prices * directions
        pnl_before_commission = position_volume * price_changes
        total_commission = position_volume * self.config.trading.taker_commission * 2
        final_pnls = pnl_before_commission - total_commission

        # Записываем результаты
        for i, trade in enumerate(self.trades):
            trade['pnl'] = final_pnls[i]
            trade['commission'] = total_commission

        self.total_pnl = final_pnls.sum()
        self._create_equity_curve_minimal()

    def _create_equity_curve_minimal(self) -> None:
        """Минимальная equity curve для оптимизации"""
        if not self.trades:
            self._create_equity_curve_empty()
            return

        current_capital = self.config.trading.initial_capital + self.total_pnl

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
                'equity': current_capital,
                'price': self.ohlc_data[-1, 3],
                'total_pnl': self.total_pnl,
                'unrealized_pnl': 0.0
            }
        ]

    def _create_equity_curve_empty(self) -> None:
        """Пустая equity curve"""
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
        """Выводит результаты"""
        execution_time = self.end_time - self.start_time
        print("-" * 60)
        print("РЕЗУЛЬТАТЫ ALL БЭКТЕСТА")
        print("-" * 60)
        print(f"Время выполнения: {execution_time:.2f} секунд")
        print(f"Обработано баров: {len(self.timestamps):,}")
        print(f"Скорость: {len(self.timestamps) / execution_time:,.0f} баров/сек")

    def get_results(self) -> Dict[str, Any]:
        """Возвращает результаты"""
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