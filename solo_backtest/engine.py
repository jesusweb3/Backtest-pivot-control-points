# solo_backtest/engine.py

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
import time

from settings.settings import BacktestConfig


class SoloBacktestEngine:
    """
    Движок для единичного бэктестинга - оригинальная проверенная логика.
    Без оптимизаций, максимальная точность и совместимость с TradingView.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

        # Данные
        self.data = None
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
        """Запускает полный цикл бэктестирования"""
        try:
            self.start_time = time.time()

            # 1. Загружаем данные
            if not self._load_data():
                return False

            # 2. Рассчитываем пивоты (оригинальная логика)
            self._calculate_pivots_original()

            # 3. Симулируем стратегию
            self._simulate_strategy()

            # 4. Рассчитываем результаты
            self._calculate_results()

            self.end_time = time.time()

            if not quiet_mode:
                self._print_results()

            return True

        except Exception as e:
            if not quiet_mode:
                print(f"Ошибка solo бэктестирования: {e}")
                import traceback
                traceback.print_exc()
            return False

    def _load_data(self) -> bool:
        """Загружает данные из CSV файла"""
        try:
            if not os.path.exists(self.config.data.csv_path):
                print(f"Ошибка: Файл не найден: {self.config.data.csv_path}")
                return False

            # Загружаем с pandas
            self.data = pd.read_csv(self.config.data.csv_path)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)

            # Создаем numpy массивы для совместимости
            self.timestamps = self.data['timestamp'].values
            self.ohlc_data = np.column_stack([
                self.data['open'].values,
                self.data['high'].values,
                self.data['low'].values,
                self.data['close'].values
            ]).astype(np.float64)

            print(f"Загружено {len(self.data)} баров")
            print(f"Период: {self.data.iloc[0]['timestamp']} - {self.data.iloc[-1]['timestamp']}")

            return True

        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return False

    def _calculate_pivots_original(self) -> None:
        """
        Оригинальная логика расчета пивотов - точное соответствие TradingView
        """
        left_bars = self.config.pivot.left_bars
        right_bars = self.config.pivot.right_bars

        self.pivot_highs = []
        self.pivot_lows = []

        # Проходим по всем барам, исключая границы
        for i in range(left_bars, len(self.data) - right_bars):
            current_high = self.data.iloc[i]['high']
            current_low = self.data.iloc[i]['low']

            # Проверяем pivot high
            is_pivot_high = True

            # Проверяем левую сторону
            for j in range(i - left_bars, i):
                if self.data.iloc[j]['high'] > current_high:
                    is_pivot_high = False
                    break

            # Проверяем правую сторону
            if is_pivot_high:
                for j in range(i + 1, i + right_bars + 1):
                    if self.data.iloc[j]['high'] >= current_high:
                        is_pivot_high = False
                        break

            # Если это pivot high, добавляем его
            if is_pivot_high:
                # Пивот подтверждается через right_bars баров
                confirmed_at_idx = i + right_bars
                if confirmed_at_idx < len(self.data):
                    self.pivot_highs.append({
                        'timestamp': self.data.iloc[i]['timestamp'],
                        'price': current_high,
                        'confirmed_at': self.data.iloc[confirmed_at_idx]['timestamp'],
                        'bar_index': i
                    })

            # Проверяем pivot low
            is_pivot_low = True

            # Проверяем левую сторону
            for j in range(i - left_bars, i):
                if self.data.iloc[j]['low'] < current_low:
                    is_pivot_low = False
                    break

            # Проверяем правую сторону
            if is_pivot_low:
                for j in range(i + 1, i + right_bars + 1):
                    if self.data.iloc[j]['low'] <= current_low:
                        is_pivot_low = False
                        break

            # Если это pivot low, добавляем его
            if is_pivot_low:
                confirmed_at_idx = i + right_bars
                if confirmed_at_idx < len(self.data):
                    self.pivot_lows.append({
                        'timestamp': self.data.iloc[i]['timestamp'],
                        'price': current_low,
                        'confirmed_at': self.data.iloc[confirmed_at_idx]['timestamp'],
                        'bar_index': i
                    })

        print(f"Найдено pivot highs: {len(self.pivot_highs)}")
        print(f"Найдено pivot lows: {len(self.pivot_lows)}")

    def _simulate_strategy(self) -> None:
        """
        Оригинальная симуляция стратегии - точное соответствие Pine Script
        """
        self.signals = []
        self.trades = []

        # Создаем словарь для быстрого поиска пивотов по времени подтверждения
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

        # Переменные состояния стратегии (как в Pine Script)
        hprice = 0.0
        lprice = 0.0
        le = False
        se = False
        current_position = None
        current_entry = None

        # Проходим по каждому бару
        for i, row in self.data.iterrows():
            timestamp = row['timestamp']
            current_high = row['high']
            current_low = row['low']

            # 1. Проверяем подтверждение пивотов на текущем баре
            if timestamp in pivot_confirmations:
                for pivot_event in pivot_confirmations[timestamp]:
                    if pivot_event['type'] == 'HIGH':
                        hprice = pivot_event['price']
                        le = True
                    else:  # LOW
                        lprice = pivot_event['price']
                        se = True

            # 2. Проверяем условия для входа в позицию
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

            # 3. Деактивируем условия при пробое в нашу сторону
            if le and current_high > hprice and current_position == "LONG":
                le = False
            if se and current_low < lprice and current_position == "SHORT":
                se = False

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

        # Определяем направление
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

    def _calculate_results(self) -> None:
        """Рассчитывает результаты торговли"""
        if not self.trades:
            self.total_pnl = 0.0
            self._create_equity_curve_empty()
            return

        # Рассчитываем PnL для каждой сделки
        position_volume = self.config.trading.position_size * self.config.trading.leverage

        for trade in self.trades:
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            direction = 1.0 if trade['entry_type'] == 'LONG' else -1.0

            # Расчет процентного изменения с учетом направления
            price_change = (exit_price - entry_price) / entry_price * direction

            # PnL до комиссий
            pnl_before_commission = position_volume * price_change

            # Комиссии (за вход и выход)
            commission = position_volume * self.config.trading.taker_commission * 2

            # Итоговый PnL
            final_pnl = pnl_before_commission - commission

            trade['pnl'] = final_pnl
            trade['commission'] = commission

        self.total_pnl = sum(trade['pnl'] for trade in self.trades)
        self._create_equity_curve()

    def _create_equity_curve(self) -> None:
        """Создает кривую капитала"""
        self.equity_curve = []
        current_capital = self.config.trading.initial_capital

        # Начальная точка
        self.equity_curve.append({
            'timestamp': self.data.iloc[0]['timestamp'],
            'equity': current_capital,
            'price': self.data.iloc[0]['close'],
            'total_pnl': 0.0,
            'unrealized_pnl': 0.0
        })

        # Точки после каждой сделки
        for trade in self.trades:
            current_capital += trade['pnl']

            # Находим цену закрытия на момент выхода из сделки
            exit_row = self.data[self.data['timestamp'] == trade['exit_time']]
            close_price = exit_row.iloc[0]['close'] if not exit_row.empty else trade['exit_price']

            self.equity_curve.append({
                'timestamp': trade['exit_time'],
                'equity': current_capital,
                'price': close_price,
                'total_pnl': current_capital - self.config.trading.initial_capital,
                'unrealized_pnl': 0.0
            })

        # Финальная точка
        if self.equity_curve[-1]['timestamp'] != self.data.iloc[-1]['timestamp']:
            self.equity_curve.append({
                'timestamp': self.data.iloc[-1]['timestamp'],
                'equity': current_capital,
                'price': self.data.iloc[-1]['close'],
                'total_pnl': current_capital - self.config.trading.initial_capital,
                'unrealized_pnl': 0.0
            })

    def _create_equity_curve_empty(self) -> None:
        """Создает пустую equity curve если нет сделок"""
        if len(self.data) == 0:
            return

        self.equity_curve = [
            {
                'timestamp': self.data.iloc[0]['timestamp'],
                'equity': self.config.trading.initial_capital,
                'price': self.data.iloc[0]['close'],
                'total_pnl': 0.0,
                'unrealized_pnl': 0.0
            },
            {
                'timestamp': self.data.iloc[-1]['timestamp'],
                'equity': self.config.trading.initial_capital,
                'price': self.data.iloc[-1]['close'],
                'total_pnl': 0.0,
                'unrealized_pnl': 0.0
            }
        ]

    def _print_results(self) -> None:
        """Выводит результаты бэктеста"""
        execution_time = self.end_time - self.start_time
        print("-" * 60)
        print("РЕЗУЛЬТАТЫ SOLO БЭКТЕСТА")
        print("-" * 60)
        print(f"Время выполнения: {execution_time:.2f} секунд")
        print(f"Обработано баров: {len(self.data):,}")
        print(f"Скорость: {len(self.data) / execution_time:,.0f} баров/сек")
        print(f"Начальный капитал: ${self.config.trading.initial_capital:,.2f}")
        print(f"Общая прибыль: ${self.total_pnl:,.2f}")
        final_capital = self.config.trading.initial_capital + self.total_pnl
        print(f"Итоговый капитал: ${final_capital:,.2f}")
        print(f"Доходность: {(final_capital / self.config.trading.initial_capital - 1) * 100:.2f}%")

    def get_results(self) -> Dict[str, Any]:
        """Возвращает результаты для аналитики"""
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
                'total_bars': len(self.data) if self.data is not None else 0,
                'total_signals': len(self.signals),
                'total_trades': len(self.trades)
            }
        }