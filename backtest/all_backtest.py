# backtest/all_backtest.py

import time
from typing import List, Optional

from settings.settings import BacktestConfig
from core.parameter_optimizer import ParameterOptimizer, OptimizationResult
from backtest.console_manager import LiveLogger


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
            # Останавливаем логирование и сохраняем лог
            self.logger.stop_logging()
            self._save_console_log()

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

        top_results = self.optimizer.get_top_results()

        if top_results:
            # Создаем сводный отчет по оптимизации
            self._export_optimization_summary(top_results)

            print(f"Экспорт завершен. Топ-{len(top_results)} результатов сохранены.")
            print(f"Метрика ранжирования: {self.config.optimization.ranking_metric}")

    def _create_optimization_visualizations(self):
        """Создает графики для лучших результатов оптимизации"""
        top_results = self.optimizer.get_top_results()

        if top_results:
            print("Создание графиков для всех топ результатов...")
            self._create_optimization_charts(top_results)

    def _export_optimization_summary(self, results: List[OptimizationResult]):
        """Экспортирует сводку результатов оптимизации в Excel"""
        try:
            import pandas as pd
            from datetime import datetime
            import os

            # Подготавливаем данные для экспорта
            summary_data = []
            for result in results:
                summary_data.append({
                    'Left_Bars': result.left_bars,
                    'Right_Bars': result.right_bars,
                    'Total_Return_%': result.total_return,
                    'Total_PnL_$': result.total_pnl,
                    'Max_Drawdown_%': result.max_drawdown_percent,
                    'Sharpe_Ratio': result.sharpe_ratio,
                    'Profit_Factor': result.profit_factor,
                    'Win_Rate_%': result.win_rate,
                    'Total_Trades': result.total_trades,
                    'Calmar_Ratio': result.calmar_ratio,
                    'Sortino_Ratio': result.sortino_ratio,
                    'Avg_Trade_$': result.avg_trade,
                    'Expectancy_$': result.expectancy,
                    'Execution_Time_s': result.execution_time,
                    'Results_Path': result.results_path
                })

            df = pd.DataFrame(summary_data)

            # Создаем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_summary_{timestamp}.xlsx"

            # Сохраняем в корневую папку results/
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            filepath = os.path.join(results_dir, filename)

            # Сохраняем
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Optimization_Summary', index=False)

                # Добавляем статистику оптимизации
                stats_data = [
                    ('Параметр', 'Значение'),
                    ('Всего комбинаций', self.config.get_total_combinations()),
                    ('Успешных результатов', len([r for r in results if r.success])),
                    ('Прибыльных комбинаций', len([r for r in results if r.total_return > 0])),
                    ('Диапазон Left Bars',
                     f"{self.config.optimization.left_bars_range[0]}-{self.config.optimization.left_bars_range[1]}"),
                    ('Диапазон Right Bars',
                     f"{self.config.optimization.right_bars_range[0]}-{self.config.optimization.right_bars_range[1]}"),
                    ('Метрика ранжирования', self.config.optimization.ranking_metric),
                    ('Максимум процессов', self.config.optimization.max_workers or 'Авто'),
                    ('Сохранять только прибыльные', self.config.optimization.save_only_profitable)
                ]

                stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
                stats_df.to_excel(writer, sheet_name='Optimization_Stats', index=False)

            print(f"Сводка оптимизации сохранена: {filepath}")

        except Exception as e:
            print(f"Ошибка экспорта сводки: {e}")

    def _create_optimization_charts(self, top_results: List[OptimizationResult]):
        """Создает графики для лучших результатов оптимизации параллельно с прогресс-баром"""
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
        Создает графики для одного результата оптимизации
        ВАЖНО: функция должна быть на верхнем уровне модуля для pickle
        """
        import io
        import sys
        old_stdout = None

        try:
            from settings.settings import BacktestConfig
            from core.backtest_engine import BacktestEngine
            from analytics.visualizer import BacktestVisualizer

            # Воссоздаем конфигурацию для этого результата
            result_config = BacktestConfig.from_dict(config_dict)
            result_config.pivot.left_bars = result.left_bars
            result_config.pivot.right_bars = result.right_bars
            result_config.optimization.enable_optimization = False  # Отключаем оптимизацию

            # Запускаем бэктест для получения данных
            engine = BacktestEngine(result_config)
            backtest_success = engine.run_backtest(quiet_mode=True)

            if not backtest_success:
                return False

            results_data = engine.get_results()

            # Заглушаем вывод при создании графиков
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

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

    def _save_console_log(self):
        """Сохраняет консольный лог оптимизации в файл"""
        try:
            log_content = self.logger.get_log_content()
            if log_content and self.optimizer:
                # Создаем специальный лог для оптимизации в корневой папке results/
                import os
                from datetime import datetime

                results_dir = "results"
                os.makedirs(results_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"optimization_log_{timestamp}.txt"
                log_filepath = os.path.join(results_dir, log_filename)

                with open(log_filepath, 'w', encoding='utf-8') as f:
                    f.write(log_content)

                print(f"Лог оптимизации сохранен: {log_filepath}")

        except Exception as e:
            print(f"Не удалось сохранить лог оптимизации: {e}")

    @staticmethod
    def _print_success():
        """Выводит сообщение об успешном завершении"""
        print("\n" + "=" * 60)
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print("=" * 60)