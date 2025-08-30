# core/backtest_engine.py

import pandas as pd
import os
from typing import List, Dict, Any
import time

from config.settings import BacktestConfig


class BacktestEngine:
    """
    Простой и эффективный движок бэктестинга на основе рабочего тестового модуля.
    Реализует точную логику TradingView без сложной событийной системы.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.bars: List[Dict] = []
        self.pivot_highs: List[Dict] = []
        self.pivot_lows: List[Dict] = []
        self.signals: List[Dict] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # Статистика выполнения
        self.start_time = None
        self.end_time = None
        self.total_pnl = 0.0

    def run_backtest(self) -> bool:
        """
        Запускает полный цикл бэктестирования
        """
        try:
            self.start_time = time.time()

            # 1. Загружаем данные
            if not self._load_data():
                return False

            # 2. Рассчитываем пивоты
            self._calculate_pivots()

            # 3. Генерируем сигналы и сделки
            self._simulate_strategy()

            # 4. Рассчитываем PnL и создаем equity curve
            self._calculate_results()

            self.end_time = time.time()
            self._print_results()

            return True

        except Exception as e:
            print(f"Ошибка бэктестирования: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_data(self) -> bool:
        """Загружает данные из CSV файла"""
        try:
            if not os.path.exists(self.config.data.csv_path):
                print(f"Ошибка: Файл не найден: {self.config.data.csv_path}")
                return False

            # Загружаем CSV
            data = pd.read_csv(self.config.data.csv_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp').reset_index(drop=True)

            # Конвертируем в список баров
            self.bars = []
            for _, row in data.iterrows():
                self.bars.append({
                    'timestamp': row['timestamp'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })

            print(f"Загружен период: {self.bars[0]['timestamp']} - {self.bars[-1]['timestamp']}")

            return True

        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return False

    def _calculate_pivots(self) -> None:
        """
        Рассчитывает пивоты - ТОЧНАЯ КОПИЯ из test_tradingview_logic.py
        """
        self.pivot_highs = []
        self.pivot_lows = []

        left_bars = self.config.pivot.left_bars
        right_bars = self.config.pivot.right_bars
        min_bars_needed = left_bars + right_bars + 1

        for i in range(min_bars_needed, len(self.bars)):
            # Индекс кандидатной свечи (R баров назад от текущей)
            candidate_idx = i - right_bars
            candidate_bar = self.bars[candidate_idx]

            # Проверяем PIVOT HIGH
            is_pivot_high = True
            candidate_high = candidate_bar['high']

            # Проверяем L баров слева от кандидата
            for j in range(candidate_idx - left_bars, candidate_idx):
                if j >= 0 and self.bars[j]['high'] >= candidate_high:
                    is_pivot_high = False
                    break

            # Проверяем R баров справа от кандидата
            if is_pivot_high:
                for j in range(candidate_idx + 1, i + 1):
                    if j < len(self.bars) and self.bars[j]['high'] > candidate_high:
                        is_pivot_high = False
                        break

            if is_pivot_high:
                self.pivot_highs.append({
                    'timestamp': candidate_bar['timestamp'],
                    'price': candidate_high,
                    'confirmed_at': self.bars[i]['timestamp'],
                    'bar_index': candidate_idx
                })

            # Проверяем PIVOT LOW
            is_pivot_low = True
            candidate_low = candidate_bar['low']

            # Проверяем L баров слева от кандидата
            for j in range(candidate_idx - left_bars, candidate_idx):
                if j >= 0 and self.bars[j]['low'] <= candidate_low:
                    is_pivot_low = False
                    break

            # Проверяем R баров справа от кандидата
            if is_pivot_low:
                for j in range(candidate_idx + 1, i + 1):
                    if j < len(self.bars) and self.bars[j]['low'] < candidate_low:
                        is_pivot_low = False
                        break

            if is_pivot_low:
                self.pivot_lows.append({
                    'timestamp': candidate_bar['timestamp'],
                    'price': candidate_low,
                    'confirmed_at': self.bars[i]['timestamp'],
                    'bar_index': candidate_idx
                })

    def _simulate_strategy(self) -> None:
        """
        Симулирует стратегию - ТОЧНАЯ КОПИЯ из test_tradingview_logic.py
        """
        self.signals = []
        self.trades = []

        # Глобальное состояние стратегии
        hprice = 0.0
        lprice = 0.0
        le = False
        se = False

        # Текущая позиция
        current_position = None
        current_entry = None

        # Создаем словарь пивотов по времени подтверждения
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

        # Проходим через ВСЕ бары подряд
        for bar in self.bars:
            current_time = bar['timestamp']
            current_high = bar['high']
            current_low = bar['low']

            # 1. СНАЧАЛА проверяем подтверждение новых пивотов на этом баре
            if current_time in pivot_confirmations:
                for pivot_event in pivot_confirmations[current_time]:
                    if pivot_event['type'] == 'HIGH':
                        hprice = pivot_event['price']
                        le = True
                    else:  # LOW
                        lprice = pivot_event['price']
                        se = True

            # 2. ЗАТЕМ проверяем сигналы на этом же баре
            if le and 0 < hprice < current_high and current_position != "LONG":
                entry_price = hprice + self.config.trading.min_tick

                # Если была SHORT позиция - закрываем её
                if current_position == "SHORT":
                    position_volume = self.config.trading.position_size * self.config.trading.leverage
                    trade = {
                        'entry_time': current_entry['time'],
                        'entry_price': current_entry['price'],
                        'entry_type': 'SHORT',
                        'exit_time': current_time,
                        'exit_price': entry_price,
                        'exit_type': 'LONG_SIGNAL',
                        'quantity': position_volume,
                        'symbol': self.config.data.symbol,
                        'direction': 'SHORT'
                    }
                    self.trades.append(trade)

                # Открываем LONG позицию
                self.signals.append({
                    'timestamp': current_time,
                    'type': 'LONG',
                    'price': entry_price
                })

                current_position = "LONG"
                current_entry = {'time': current_time, 'price': entry_price}
                le = False

            # SHORT сигнал (пробой lprice вниз)
            elif se and lprice > 0 and current_low < lprice and current_position != "SHORT":
                entry_price = lprice - self.config.trading.min_tick

                # Если была LONG позиция - закрываем её
                if current_position == "LONG":
                    position_volume = self.config.trading.position_size * self.config.trading.leverage
                    trade = {
                        'entry_time': current_entry['time'],
                        'entry_price': current_entry['price'],
                        'entry_type': 'LONG',
                        'exit_time': current_time,
                        'exit_price': entry_price,
                        'exit_type': 'SHORT_SIGNAL',
                        'quantity': position_volume,
                        'symbol': self.config.data.symbol,
                        'direction': 'LONG'
                    }
                    self.trades.append(trade)

                # Открываем SHORT позицию
                self.signals.append({
                    'timestamp': current_time,
                    'type': 'SHORT',
                    'price': entry_price
                })

                current_position = "SHORT"
                current_entry = {'time': current_time, 'price': entry_price}
                se = False

            # 3. Деактивируем условия если был пробой (но позиция уже есть)
            if le and current_high > hprice and current_position == "LONG":
                le = False

            if se and current_low < lprice and current_position == "SHORT":
                se = False

        print(f"Найдено pivot highs: {len(self.pivot_highs)}")
        print(f"Найдено pivot lows: {len(self.pivot_lows)}")
        print(f"Сгенерировано сигналов: {len(self.signals)}")
        print(f"Совершено сделок: {len(self.trades)}")

    def _calculate_results(self) -> None:
        """Рассчитывает финальные результаты и создает equity curve"""
        self.total_pnl = 0.0
        position_volume = self.config.trading.position_size * self.config.trading.leverage

        for trade in self.trades:
            pnl = self._calculate_trade_pnl(trade)
            trade['pnl'] = pnl
            trade['commission'] = self._calculate_trade_commission(position_volume)
            self.total_pnl += pnl

        self._create_equity_curve()

    def _calculate_trade_pnl(self, trade: Dict) -> float:

        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        entry_type = trade['entry_type']

        # Объем позиции с учетом плеча
        position_volume = self.config.trading.position_size * self.config.trading.leverage

        # Расчет изменения цены в процентах
        if entry_type == 'LONG':
            price_change_percent = (exit_price - entry_price) / entry_price
        else:  # SHORT
            price_change_percent = (entry_price - exit_price) / entry_price

        # PnL от изменения цены
        pnl_before_commission = position_volume * price_change_percent

        # Комиссия за вход и выход
        total_commission = position_volume * self.config.trading.taker_commission * 2

        # Итоговый PnL
        final_pnl = pnl_before_commission - total_commission

        return final_pnl

    def _calculate_trade_commission(self, position_volume: float) -> float:
        """Рассчитывает комиссию за сделку"""
        return position_volume * self.config.trading.taker_commission * 2

    def _create_equity_curve(self) -> None:
        """Создает кривую капитала"""
        current_capital = self.config.trading.initial_capital
        self.equity_curve = []

        # Добавляем начальную точку
        if self.bars:
            self.equity_curve.append({
                'timestamp': self.bars[0]['timestamp'],
                'equity': current_capital,
                'price': self.bars[0]['close'],
                'total_pnl': 0.0,
                'unrealized_pnl': 0.0
            })

        # Добавляем точки после каждой сделки
        for trade in self.trades:
            current_capital += trade['pnl']

            # Найдем соответствующий бар для получения цены
            trade_bar = None
            for bar in self.bars:
                if bar['timestamp'] == trade['exit_time']:
                    trade_bar = bar
                    break

            if trade_bar:
                self.equity_curve.append({
                    'timestamp': trade['exit_time'],
                    'equity': current_capital,
                    'price': trade_bar['close'],
                    'total_pnl': current_capital - self.config.trading.initial_capital,
                    'unrealized_pnl': 0.0
                })

        # Добавляем финальную точку
        if self.bars and self.equity_curve:
            final_bar = self.bars[-1]
            if self.equity_curve[-1]['timestamp'] != final_bar['timestamp']:
                self.equity_curve.append({
                    'timestamp': final_bar['timestamp'],
                    'equity': current_capital,
                    'price': final_bar['close'],
                    'total_pnl': current_capital - self.config.trading.initial_capital,
                    'unrealized_pnl': 0.0
                })

    def _print_results(self) -> None:
        """Выводит результаты бэктеста"""
        print("-" * 60)
        print("РЕЗУЛЬТАТЫ БЭКТЕСТА")
        print("-" * 60)
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
            'config': self.config.to_dict(),
            'total_pnl': self.total_pnl,
            'execution_stats': {
                'execution_time': self.end_time - self.start_time if self.end_time else 0,
                'total_bars': len(self.bars),
                'total_signals': len(self.signals),
                'total_trades': len(self.trades)
            }
        }