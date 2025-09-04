# all_backtest/runner.py

import time
import sys
from io import StringIO
from typing import List, Optional

from settings.settings import BacktestConfig
from all_backtest.optimizer import ParameterOptimizer, OptimizationResult
from all_backtest.recorder import TradeRecorder


class LiveLogger:
    """
    Простой логгер для дублирования живого вывода в буфер.
    Пользователь видит весь прогресс в реальном времени,
    а мы сохраняем точную копию в лог-файл.
    """

    def __init__(self):
        self.log_buffer = StringIO()
        self.original_stdout = sys.stdout
        self.tee_output: Optional['TeeOutput'] = None

    def start_logging(self):
        """Начинает дублирование вывода в буфер и на экран"""
        self.tee_output = TeeOutput(self.original_stdout, self.log_buffer)
        sys.stdout = self.tee_output

    def stop_logging(self):
        """Останавливает логирование и возвращает оригинальный stdout"""
        sys.stdout = self.original_stdout

    def get_log_content(self) -> str:
        """Возвращает весь перехваченный лог"""
        return self.log_buffer.getvalue()


class TeeOutput:
    """Простая обертка для дублирования вывода в несколько потоков"""

    def __init__(self, console_file, log_file):
        self.console_file = console_file  # sys.stdout для экрана
        self.log_file = log_file  # StringIO для буфера

    def write(self, text):
        """Записывает текст и на экран и в буфер"""
        self.console_file.write(text)
        self.log_file.write(text)

    def flush(self):
        """Очищает буферы"""
        self.console_file.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        """Проксирует все остальные методы к console_file"""
        return getattr(self.console_file, name)


class AllBacktestRunner:
    """
    Оркестратор параметрической оптимизации (множественный бэктест всех комбинаций).
    Управляет полным пайплайном: оптимизация → анализ → экспорт → визуализация.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = LiveLogger()
        self.optimizer: Optional[ParameterOptimizer] = None
        self.optimization_results: List[OptimizationResult] = []
        self.execution_time: float = 0.0

    def run(self) -> bool:
        """Запускает полный пайплайн параметрической оптимизации"""

        # Начинаем логирование
        self.logger.start_logging()

        try:
            self._print_header()

            # Инициализируем оптимизатор
            self.optimizer = ParameterOptimizer(self.config)

            # Выполняем оптимизацию
            if not self._execute_optimization():
                return False

            # Показываем топ результатов
            self._print_top_results()

            # Экспортируем результаты
            self._export_optimization_results()

            # Создаем графики для топ результатов
            self._create_optimization_visualizations()

            self._print_success()

            return True

        except KeyboardInterrupt:
            print("\nВыполнение прервано пользователем")
            return False
        except Exception as e:
            print(f"\nОшибка выполнения оптимизации: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Останавливаем логирование (без сохранения лога)
            self.logger.stop_logging()

    @staticmethod
    def _print_header():
        """Выводит заголовок оптимизации"""
        print("=" * 80)
        print("ПАРАМЕТРИЧЕСКАЯ ОПТИМИЗАЦИЯ СТРАТЕГИИ")
        print("=" * 80)

    def _execute_optimization(self) -> bool:
        """Выполняет параметрическую оптимизацию"""
        try:
            start_time = time.time()

            # Запускаем оптимизацию
            self.optimization_results = self.optimizer.run_optimization()

            self.execution_time = time.time() - start_time

            if not self.optimization_results:
                print("Оптимизация не дала результатов")
                return False

            return True

        except Exception as e:
            print(f"Ошибка оптимизации: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _print_top_results(self):
        """Показывает топ результатов в консоли"""
        if self.optimizer:
            print()
            self.optimizer.print_top_results(top_n=20)

    def _export_optimization_results(self):
        """Экспортирует результаты оптимизации"""
        print()
        print("-" * 60)
        print("ЭКСПОРТ РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
        print("-" * 60)

        # Получаем ВСЕ результаты для сводки (включая убыточные)
        all_results = self.optimizer.get_all_results_for_export()

        if all_results:
            # Экспортируем сводку со ВСЕМИ результатами
            dict_results = []
            for result in all_results:
                dict_results.append({
                    'left_bars': result.left_bars,
                    'right_bars': result.right_bars,
                    'success': result.success,
                    'total_return': result.total_return,
                    'total_pnl': result.total_pnl,
                    'max_drawdown_percent': result.max_drawdown_percent,
                    'sharpe_ratio': result.sharpe_ratio,
                    'profit_factor': result.profit_factor,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'sortino_ratio': result.sortino_ratio,
                    'recovery_factor': result.recovery_factor,
                    'avg_trade': result.avg_trade,
                    'expectancy': result.expectancy,
                    'execution_time': result.execution_time,
                    'error_message': result.error_message
                })

            TradeRecorder.export_optimization_summary(
                optimization_results=dict_results,
                config=self.config.to_dict()
            )

            profitable_count = len([r for r in all_results if r.total_return > 0])
            print(f"Экспорт завершен. Всего результатов: {len(all_results)}")
            print(f"Прибыльных комбинаций: {profitable_count}")
            print(f"Метрика ранжирования: {self.config.optimization.ranking_metric}")
        else:
            print("Нет результатов для экспорта")

    def _create_optimization_visualizations(self):
        """Создает графики для лучших результатов оптимизации"""
        top_results = self.optimizer.get_top_results()

        if top_results:
            print("Создание графиков для всех топ результатов...")
            self._create_optimization_charts(top_results)

    def _create_optimization_charts(self, top_results: List[OptimizationResult]):
        """Создает графики для лучших результатов оптимизации"""
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            import time

            max_workers = min(mp.cpu_count(), len(top_results))
            total_charts = len(top_results)
            completed_charts = 0
            failed_charts = 0
            start_time = time.time()

            print(f"Создание {total_charts} графиков на {max_workers} процессах...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Отправляем все задачи на выполнение
                future_to_result = {
                    executor.submit(self._create_single_chart, result, self.config.to_dict()): result
                    for result in top_results
                }

                # Собираем результаты по мере завершения
                for future in as_completed(future_to_result):
                    try:
                        chart_created = future.result()
                        if chart_created:
                            completed_charts += 1
                        else:
                            failed_charts += 1
                    except (ValueError, AttributeError, TypeError, OSError):
                        failed_charts += 1

                    # Показываем прогресс
                    total_processed = completed_charts + failed_charts
                    progress_percent = (total_processed / total_charts) * 100
                    elapsed_time = time.time() - start_time
                    elapsed_minutes = elapsed_time / 60

                    # Создаем прогресс-бар
                    bar_width = 40
                    filled_width = int(bar_width * progress_percent / 100)
                    bar = '█' * filled_width + '░' * (bar_width - filled_width)

                    print(f"\r📊 [{bar}] {progress_percent:.1f}% | "
                          f"{total_processed}/{total_charts} | "
                          f"✅ {completed_charts} | "
                          f"❌ {failed_charts} | "
                          f"⏱️ {elapsed_minutes:.1f}мин",
                          end="", flush=True)

            print()  # Новая строка после завершения
            print(f"Создано графиков: {completed_charts}/{total_charts}")

        except Exception as e:
            print(f"Ошибка создания графиков оптимизации: {e}")

    @staticmethod
    def _create_single_chart(result: OptimizationResult, config_dict: dict) -> bool:
        """
        Создает графики и Excel файлы для одного результата оптимизации
        ВАЖНО: функция должна быть на верхнем уровне модуля для pickle
        """
        import io
        import sys
        old_stdout = None

        try:
            from settings.settings import BacktestConfig
            from all_backtest.engine import AllBacktestEngine
            from all_backtest.visualizer import BacktestVisualizer
            from all_backtest.analyzer import PerformanceAnalyzer
            from all_backtest.recorder import TradeRecorder

            # Воссоздаем конфигурацию для этого результата
            result_config = BacktestConfig.from_dict(config_dict)
            result_config.pivot.left_bars = result.left_bars
            result_config.pivot.right_bars = result.right_bars
            result_config.optimization.enable_optimization = False  # Отключаем оптимизацию

            # Запускаем бэктест для получения данных с all движком
            engine = AllBacktestEngine(result_config)
            backtest_success = engine.run_backtest(quiet_mode=True)

            if not backtest_success:
                return False

            results_data = engine.get_results()

            # Анализируем производительность
            analyzer = PerformanceAnalyzer(
                equity_curve=results_data['equity_curve'],
                trades=results_data['trades'],
                initial_capital=result_config.trading.initial_capital
            )
            performance_metrics = analyzer.calculate_all_metrics()

            # Заглушаем вывод при создании файлов
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            # Создаем детализированный Excel файл для каждой комбинации
            recorder = TradeRecorder()
            recorder.export_trades_to_excel(
                trades=results_data['trades'],
                performance_metrics=performance_metrics,
                config=results_data['settings'],
                filename=f"optimization_{result.left_bars}-{result.right_bars}.xlsx"
            )

            # Создаем графики в стандартной структуре папок (L-R/charts/)
            BacktestVisualizer.create_trade_analysis_chart(
                trades=results_data['trades'],
                config=results_data['settings'],
                filename=f"optimization_analysis.png"
            )

            # Создаем график месячной доходности если данных достаточно
            if len(results_data['equity_curve']) > 30:
                BacktestVisualizer.create_monthly_returns_chart(
                    equity_curve=results_data['equity_curve'],
                    config=results_data['settings'],
                    filename=f"optimization_monthly.png"
                )

            return True

        except (ValueError, AttributeError, TypeError, ImportError, OSError):
            return False
        finally:
            # Восстанавливаем вывод в любом случае
            if old_stdout is not None:
                sys.stdout = old_stdout

    @staticmethod
    def _print_success():
        """Выводит сообщение об успешном завершении"""
        print("\n" + "=" * 60)
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print("=" * 60)