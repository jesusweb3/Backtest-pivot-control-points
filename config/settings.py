# config/settings.py

from dataclasses import dataclass, field
from typing import Dict, Any


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
    csv_path: str = "data/ethusdt_2025-08-01_2025-08-27.csv"
    timeframe: str = "1m"


@dataclass
class BacktestConfig:
    """Основная конфигурация бэктеста"""
    trading: TradingConfig = field(default_factory=TradingConfig)
    pivot: PivotConfig = field(default_factory=PivotConfig)
    data: DataConfig = field(default_factory=DataConfig)

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
            }
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """Создает конфигурацию из словаря"""
        trading = TradingConfig(**config_dict.get('trading', {}))
        pivot = PivotConfig(**config_dict.get('pivot', {}))
        data = DataConfig(**config_dict.get('data', {}))

        return cls(trading=trading, pivot=pivot, data=data)


# Глобальная конфигурация по умолчанию
DEFAULT_CONFIG = BacktestConfig()