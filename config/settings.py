# config/settings.py

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


@dataclass
class TradingConfig:
    """Конфигурация торговых параметров"""
    initial_capital: float = 200.0
    leverage: float = 4.0
    position_size: float = 100.0  # Фиксированный размер позиции в USD (маржа)
    taker_commission: float = 0.001  # 0.1% комиссия тейкера за сторону
    min_tick: float = 0.01  # Минимальный тик цены


@dataclass
class PivotConfig:
    """Конфигурация параметров Pivot Reversal Strategy"""
    left_bars: int = 78  # Количество баров слева для расчета пивота
    right_bars: int = 46  # Количество баров справа для расчета пивота


@dataclass
class DataConfig:
    """Конфигурация данных"""
    symbol: str = "ETHUSDT"
    csv_path: str = "data/ethusdt_2025-08-18_2025-08-30.csv"
    timeframe: str = "1m"


@dataclass
class OptimizationConfig:
    """Конфигурация параметрической оптимизации"""
    # Переключатель режимов
    enable_optimization: bool = True  # True = оптимизация, False = обычный бэктест

    # Диапазоны параметров для оптимизации
    left_bars_range: Tuple[int, int] = (1, 80)  # Диапазон left_bars
    right_bars_range: Tuple[int, int] = (1, 80)  # Диапазон right_bars

    # Настройки выполнения
    max_workers: int = 0  # 0 = авто (все ядра), или явно указать количество
    save_only_profitable: bool = True  # Сохранять только прибыльные комбинации

    # Настройки отчетов
    top_results_count: int = 1000  # Сколько лучших результатов сохранить
    ranking_metric: str = "total_return"  # Метрика для ранжирования: total_return, sharpe_ratio, calmar_ratio, profit_factor


@dataclass
class BacktestConfig:
    """Основная конфигурация бэктеста"""
    trading: TradingConfig = field(default_factory=TradingConfig)
    pivot: PivotConfig = field(default_factory=PivotConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует конфигурацию в словарь"""
        return {
            'trading': {
                'initial_capital': self.trading.initial_capital,
                'leverage': self.trading.leverage,
                'position_size': self.trading.position_size,
                'taker_commission': self.trading.taker_commission,
                'min_tick': self.trading.min_tick
            },
            'pivot': {
                'left_bars': self.pivot.left_bars,
                'right_bars': self.pivot.right_bars
            },
            'data': {
                'symbol': self.data.symbol,
                'csv_path': self.data.csv_path,
                'timeframe': self.data.timeframe
            },
            'optimization': {
                'enable_optimization': self.optimization.enable_optimization,
                'left_bars_range': self.optimization.left_bars_range,
                'right_bars_range': self.optimization.right_bars_range,
                'max_workers': self.optimization.max_workers,
                'save_only_profitable': self.optimization.save_only_profitable,
                'top_results_count': self.optimization.top_results_count,
                'ranking_metric': self.optimization.ranking_metric
            }
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """Создает конфигурацию из словаря"""
        trading = TradingConfig(**config_dict.get('trading', {}))
        pivot = PivotConfig(**config_dict.get('pivot', {}))
        data = DataConfig(**config_dict.get('data', {}))
        optimization = OptimizationConfig(**config_dict.get('optimization', {}))

        return cls(trading=trading, pivot=pivot, data=data, optimization=optimization)

    def is_optimization_mode(self) -> bool:
        """Проверяет, включен ли режим оптимизации"""
        return self.optimization.enable_optimization

    def get_total_combinations(self) -> int:
        """Возвращает общее количество комбинаций для оптимизации"""
        left_count = self.optimization.left_bars_range[1] - self.optimization.left_bars_range[0] + 1
        right_count = self.optimization.right_bars_range[1] - self.optimization.right_bars_range[0] + 1
        return left_count * right_count


# Глобальная конфигурация по умолчанию
DEFAULT_CONFIG = BacktestConfig()