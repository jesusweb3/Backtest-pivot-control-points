# settings/settings.py

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class TradingConfig:
    """Конфигурация торговых параметров"""
    initial_capital: float
    leverage: float
    position_size: float  # Фиксированный размер позиции в USD (маржа)
    taker_commission: float  # комиссия тейкера за сторону
    min_tick: float  # Минимальный тик цены


@dataclass
class PivotConfig:
    """Конфигурация параметров Pivot Reversal Strategy"""
    left_bars: int  # Количество баров слева для расчета пивота
    right_bars: int  # Количество баров справа для расчета пивота


@dataclass
class DataConfig:
    """Конфигурация данных"""
    symbol: str
    csv_path: str
    timeframe: str


@dataclass
class OptimizationConfig:
    """Конфигурация параметрической оптимизации"""
    # Переключатель режимов
    enable_optimization: bool  # True = оптимизация, False = обычный бэктест

    # Диапазоны параметров для оптимизации
    left_bars_range: Tuple[int, int]  # Диапазон left_bars
    right_bars_range: Tuple[int, int]  # Диапазон right_bars

    # Настройки выполнения
    max_workers: int  # 0 = авто (все ядра), или явно указать количество
    save_only_profitable: bool  # Сохранять только прибыльные комбинации

    # Настройки отчетов
    top_results_count: int  # Сколько лучших результатов сохранить
    ranking_metric: str  # Метрика для ранжирования: total_return, sharpe_ratio, calmar_ratio, profit_factor


@dataclass
class BacktestConfig:
    """Основная конфигурация бэктеста"""
    trading: TradingConfig
    pivot: PivotConfig
    data: DataConfig
    optimization: OptimizationConfig

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
    def load_from_user_config(cls) -> 'BacktestConfig':
        """Загружает пользовательскую конфигурацию из корневого config.py"""
        import sys
        import os

        try:
            # Добавляем корневую директорию проекта в sys.path для импорта config.py
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)

            import config as user_config

            # Создаем конфигурацию на основе пользовательских настроек
            trading_config = TradingConfig(
                initial_capital=user_config.INITIAL_CAPITAL,
                leverage=user_config.LEVERAGE,
                position_size=user_config.POSITION_SIZE,
                taker_commission=user_config.TAKER_COMMISSION,
                min_tick=user_config.MIN_TICK
            )

            pivot_config = PivotConfig(
                left_bars=user_config.LEFT_BARS,
                right_bars=user_config.RIGHT_BARS
            )

            data_config = DataConfig(
                symbol=user_config.SYMBOL,
                csv_path=user_config.CSV_PATH,
                timeframe=user_config.TIMEFRAME
            )

            optimization_config = OptimizationConfig(
                enable_optimization=user_config.ENABLE_OPTIMIZATION,
                left_bars_range=user_config.LEFT_BARS_RANGE,
                right_bars_range=user_config.RIGHT_BARS_RANGE,
                max_workers=user_config.MAX_WORKERS,
                save_only_profitable=user_config.SAVE_ONLY_PROFITABLE,
                top_results_count=user_config.TOP_RESULTS_COUNT,
                ranking_metric=user_config.RANKING_METRIC
            )

            return cls(
                trading=trading_config,
                pivot=pivot_config,
                data=data_config,
                optimization=optimization_config
            )

        except ImportError as e:
            print(f"Ошибка загрузки config.py: {e}")
            print("Создайте файл config.py в корневой папке проекта")
            sys.exit(1)
        except AttributeError as e:
            print(f"Ошибка в config.py - отсутствует параметр: {e}")
            print("Проверьте, что все необходимые параметры определены в config.py")
            sys.exit(1)

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