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
        """Экспортирует сводку результатов оптимизации в Excel с красивым форматированием"""
        try:
            import pandas as pd
            from datetime import datetime
            import os
            from openpyxl.styles import Alignment

            # Подготавливаем данные для экспорта с русскими заголовками
            summary_data = []
            for result in results:
                summary_data.append({
                    'Левые бары': result.left_bars,
                    'Правые бары': result.right_bars,
                    'Доходность %': result.total_return,
                    'Общий PnL $': result.total_pnl,
                    'Макс просадка %': result.max_drawdown_percent,
                    'Коэфф. Шарпа': result.sharpe_ratio,
                    'Профит-фактор': result.profit_factor,
                    'Винрейт %': result.win_rate,
                    'Всего сделок': result.total_trades,
                    'Средняя сделка $': result.avg_trade,
                    'Путь к результатам': result.results_path
                })

            df = pd.DataFrame(summary_data)

            # Создаем имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_summary_{timestamp}.xlsx"

            # Сохраняем в корневую папку results/
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            filepath = os.path.join(results_dir, filename)

            # Сохраняем с красивым форматированием
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Лист с результатами
                df.to_excel(writer, sheet_name='Результаты_Оптимизации', index=False)
                worksheet = writer.sheets['Результаты_Оптимизации']

                # Красивое форматирование как в solo backtest
                self._format_optimization_summary_sheet(worksheet)

                # Добавляем статистику оптимизации
                stats_data = [
                    ('Параметр', 'Значение'),
                    ('Всего комбинаций', self.config.get_total_combinations()),
                    ('Успешных результатов', len([r for r in results if r.success])),
                    ('Прибыльных комбинаций', len([r for r in results if r.total_return > 0])),
                    ('Диапазон левых баров',
                     f"{self.config.optimization.left_bars_range[0]}-{self.config.optimization.left_bars_range[1]}"),
                    ('Диапазон правых баров',
                     f"{self.config.optimization.right_bars_range[0]}-{self.config.optimization.right_bars_range[1]}"),
                    ('Метрика ранжирования', self.config.optimization.ranking_metric),
                    ('Максимум процессов', self.config.optimization.max_workers or 'Авто'),
                    ('Сохранять только прибыльные', self.config.optimization.save_only_profitable)
                ]

                stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
                stats_df.to_excel(writer, sheet_name='Статистика_Оптимизации', index=False)

                # Форматируем лист статистики
                stats_worksheet = writer.sheets['Статистика_Оптимизации']
                self._format_stats_sheet(stats_worksheet)

            print(f"Сводка оптимизации сохранена: {filepath}")

        except Exception as e:
            print(f"Ошибка экспорта сводки: {e}")

    @staticmethod
    def _format_optimization_summary_sheet(worksheet) -> None:
        """Форматирует лист с результатами оптимизации"""
        from openpyxl.styles import Alignment

        # Закрепляем первую строку (заголовки)
        worksheet.freeze_panes = 'A2'

        # Автоматическая настройка ширины колонок
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    cell_length = len(str(cell.value)) if cell.value else 0
                    if cell_length > max_length:
                        max_length = cell_length
                except (AttributeError, TypeError):
                    pass

            # Устанавливаем ширину с запасом, но не более 60 символов
            adjusted_width = min(max_length + 2, 60)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Выравнивание по центру для всех ячеек
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(horizontal='center', vertical='center')

    @staticmethod
    def _format_stats_sheet(worksheet) -> None:
        """Форматирует лист со статистикой"""
        from openpyxl.styles import Alignment

        # Настраиваем ширину колонок
        worksheet.column_dimensions['A'].width = 25
        worksheet.column_dimensions['B'].width = 20

        # Выравнивание
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(horizontal='center', vertical='center')

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
            from core.all_backtest_engine import AllBacktestEngine
            from analytics.visualizer import BacktestVisualizer
            from analytics.performance_analyzer import PerformanceAnalyzer
            from analytics.trade_recorder import TradeRecorder

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