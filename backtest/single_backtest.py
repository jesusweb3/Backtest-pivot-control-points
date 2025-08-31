# backtest/single_backtest.py

import time
from typing import Dict, Any

from settings.settings import BacktestConfig
from core.backtest_engine import BacktestEngine
from analytics.performance_analyzer import PerformanceAnalyzer
from analytics.trade_recorder import TradeRecorder
from analytics.visualizer import BacktestVisualizer
from backtest.console_manager import LiveLogger


class SingleBacktestRunner:
    """
    Оркестратор единичного бэктеста.
    Управляет полным пайплайном: выполнение → анализ → экспорт → визуализация.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = LiveLogger()
        self.results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.execution_time: float = 0.0

    def run(self) -> bool:
        """Запускает полный пайплайн единичного бэктеста"""

        self.logger.start_logging()

        try:
            self._print_header()
            self._print_configuration()

            # Выполняем бэктест
            if not self._execute_backtest():
                return False

            # Анализируем результаты
            self._analyze_performance()

            # Выводим результаты
            self._print_results()

            # Экспортируем результаты
            self._export_results()

            # Создаем визуализации
            self._create_visualizations()

            self._print_success()

            return True

        except KeyboardInterrupt:
            print("\nВыполнение прервано пользователем")
            return False
        except Exception as e:
            print(f"\nОшибка выполнения единичного бэктеста: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.logger.stop_logging()
            self._save_console_log()

    @staticmethod
    def _print_header():
        """Выводит заголовок единичного бэктеста"""
        print("-" * 60)
        print("ЗАПУСК ЕДИНИЧНОГО БЭКТЕСТА")
        print("-" * 60)

    def _print_configuration(self):
        """Выводит конфигурацию бэктеста"""
        print(f"Конфигурация:")
        print(f"  Символ: {self.config.data.symbol}")
        print(f"  Начальный капитал: ${self.config.trading.initial_capital:,.2f}")
        print(f"  Размер маржи: ${self.config.trading.position_size:,.2f}")
        print(f"  Плечо: {self.config.trading.leverage}x")
        print(f"  Комиссия: {self.config.trading.taker_commission * 100:.3f}%")
        print(f"  Pivot параметры: L={self.config.pivot.left_bars}, R={self.config.pivot.right_bars}")
        print()

    def _execute_backtest(self) -> bool:
        """Выполняет бэктест"""
        start_time = time.time()

        engine = BacktestEngine(self.config)
        backtest_success = engine.run_backtest()

        if not backtest_success:
            print("Ошибка: Бэктест не выполнен")
            return False

        self.results = engine.get_results()
        self.execution_time = time.time() - start_time

        return True

    def _analyze_performance(self):
        """Анализирует производительность стратегии"""
        analyzer = PerformanceAnalyzer(
            equity_curve=self.results['equity_curve'],
            trades=self.results['trades'],
            initial_capital=self.config.trading.initial_capital
        )

        self.performance_metrics = analyzer.calculate_all_metrics()

    def _print_results(self):
        """Выводит основные результаты бэктеста"""
        winning_trades = len([t for t in self.results['trades'] if t['pnl'] > 0])
        losing_trades = len([t for t in self.results['trades'] if t['pnl'] <= 0])
        total_commission = sum(t['commission'] for t in self.results['trades'])

        print(f"Максимальная просадка: {self.performance_metrics['max_drawdown_percent']:.2f}%")
        print(f"Винрейт: {winning_trades / max(1, len(self.results['trades'])) * 100:.1f}%")
        print(f"Всего сделок: {len(self.results['trades'])}")
        print(f"Прибыльные: {winning_trades}")
        print(f"Убыточные: {losing_trades}")
        print(f"Общие комиссии: ${total_commission:.2f}")
        print()
        print(f"Коэффициент Шарпа: {self.performance_metrics['sharpe_ratio']:.3f}")
        print(f"Профит-фактор: {self.performance_metrics['profit_factor']:.3f}")
        print(f"Средняя сделка: ${self.performance_metrics['avg_trade']:.2f}")

    def _export_results(self):
        """Экспортирует результаты в Excel"""
        print("-" * 60)
        print("ЭКСПОРТ РЕЗУЛЬТАТОВ")
        print("-" * 60)

        recorder = TradeRecorder()
        recorder.export_trades_to_excel(
            trades=self.results['trades'],
            performance_metrics=self.performance_metrics,
            config=self.results['settings']
        )

    def _create_visualizations(self):
        """Создает графики и визуализации"""
        print("Создание графиков...")
        BacktestVisualizer.create_complete_report(
            equity_curve=self.results['equity_curve'],
            trades=self.results['trades'],
            config=self.results['settings']
        )

    def _save_console_log(self):
        """Сохраняет консольный лог в файл"""
        try:
            log_content = self.logger.get_log_content()
            if log_content:  # Сохраняем только если есть содержимое
                recorder = TradeRecorder()
                recorder.save_console_log(log_content, self.results.get('settings', self.config.to_dict()))
        except Exception as e:
            print(f"Не удалось сохранить лог: {e}")

    @staticmethod
    def _print_success():
        """Выводит сообщение об успешном завершении"""
        print("\n" + "=" * 60)
        print("ВЫПОЛНЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("=" * 60)