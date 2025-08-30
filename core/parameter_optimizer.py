# core/parameter_optimizer.py

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import multiprocessing as mp

from core.backtest_engine import BacktestEngine
from analytics.performance_analyzer import PerformanceAnalyzer
from config.settings import BacktestConfig


@dataclass
class OptimizationTask:
    """Задача для оптимизации одной комбинации параметров"""
    left_bars: int
    right_bars: int
    task_id: int
    csv_path: str
    config_dict: Dict[str, Any]


@dataclass
class OptimizationResult:
    """Результат оптимизации одной комбинации"""
    left_bars: int
    right_bars: int
    task_id: int
    success: bool

    # Ключевые метрики для ранжирования
    total_return: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    calmar_ratio: float = 0.0
    execution_time: float = 0.0

    # Дополнительные метрики
    sortino_ratio: float = 0.0
    recovery_factor: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0

    # Путь к сохраненным результатам
    results_path: str = ""

    error_message: str = ""


def run_single_optimization(task: OptimizationTask) -> OptimizationResult:
    """
    Функция для выполнения одной задачи оптимизации в отдельном процессе
    ВАЖНО: эта функция должна быть на верхнем уровне модуля для pickle
    """
    try:
        start_time = time.time()

        # Создаем конфигурацию для этой комбинации параметров
        config = BacktestConfig.from_dict(task.config_dict)
        config.pivot.left_bars = task.left_bars
        config.pivot.right_bars = task.right_bars

        # Запускаем бэктест в тихом режиме (без вывода в консоль)
        engine = BacktestEngine(config)
        backtest_success = engine.run_backtest(quiet_mode=True)

        if not backtest_success:
            return OptimizationResult(
                left_bars=task.left_bars,
                right_bars=task.right_bars,
                task_id=task.task_id,
                success=False,
                error_message="Backtest failed"
            )

        # Получаем результаты
        results = engine.get_results()

        # Анализируем производительность
        analyzer = PerformanceAnalyzer(
            equity_curve=results['equity_curve'],
            trades=results['trades'],
            initial_capital=config.trading.initial_capital
        )

        performance_metrics = analyzer.calculate_all_metrics()
        execution_time = time.time() - start_time

        # Сохраняем результаты (опционально для лучших)
        results_path = ""
        save_condition = (performance_metrics['total_return'] > 0 if
                          task.config_dict['optimization']['save_only_profitable']
                          else True)

        if save_condition:
            import io
            import sys
            old_stdout = None

            try:
                # Заглушаем вывод при экспорте
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()

                from analytics.trade_recorder import TradeRecorder
                recorder = TradeRecorder()

                # Создаем уникальное имя файла
                filename = f"optimization_{task.left_bars}-{task.right_bars}.xlsx"
                results_path = recorder.export_trades_to_excel(
                    trades=results['trades'],
                    performance_metrics=performance_metrics,
                    config=results['config'],
                    filename=filename
                )

            except (ImportError, OSError, PermissionError):
                # Если не удалось сохранить - не критично
                pass
            finally:
                # Восстанавливаем вывод в любом случае
                if old_stdout is not None:
                    sys.stdout = old_stdout

        return OptimizationResult(
            left_bars=task.left_bars,
            right_bars=task.right_bars,
            task_id=task.task_id,
            success=True,
            total_return=performance_metrics['total_return'],
            total_pnl=performance_metrics.get('total_pnl', results['total_pnl']),
            max_drawdown_percent=performance_metrics['max_drawdown_percent'],
            sharpe_ratio=performance_metrics['sharpe_ratio'],
            profit_factor=performance_metrics['profit_factor'],
            win_rate=performance_metrics['win_rate'],
            total_trades=performance_metrics['total_trades'],
            calmar_ratio=performance_metrics['calmar_ratio'],
            sortino_ratio=performance_metrics['sortino_ratio'],
            recovery_factor=performance_metrics['recovery_factor'],
            avg_trade=performance_metrics['avg_trade'],
            expectancy=performance_metrics['expectancy'],
            execution_time=execution_time,
            results_path=results_path
        )

    except (ValueError, AttributeError, TypeError) as e:
        return OptimizationResult(
            left_bars=task.left_bars,
            right_bars=task.right_bars,
            task_id=task.task_id,
            success=False,
            error_message=str(e)
        )


class ParameterOptimizer:
    """
    Параметрический оптимизатор с multiprocessing для поиска лучших комбинаций
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.max_workers = config.optimization.max_workers or mp.cpu_count()
        self.optimization_results: List[OptimizationResult] = []

        # Статистика выполнения
        self.start_time = None
        self.end_time = None
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0

    def run_optimization(self, progress_callback=None) -> List[OptimizationResult]:
        """
        Запускает параметрическую оптимизацию согласно конфигурации
        """
        if not self.config.is_optimization_mode():
            raise ValueError("Режим оптимизации не включен в конфигурации")

        print("=" * 80)
        print("ПАРАМЕТРИЧЕСКАЯ ОПТИМИЗАЦИЯ СТРАТЕГИИ")
        print("=" * 80)

        # Получаем диапазоны из конфигурации
        left_range = self.config.optimization.left_bars_range
        right_range = self.config.optimization.right_bars_range

        # Генерируем все комбинации параметров
        tasks = self._generate_optimization_tasks(left_range, right_range)
        self.total_tasks = len(tasks)

        print(f"Конфигурация оптимизации:")
        print(f"  Left bars диапазон: {left_range[0]}-{left_range[1]}")
        print(f"  Right bars диапазон: {right_range[0]}-{right_range[1]}")
        print(f"  Всего комбинаций: {self.total_tasks:,}")
        print(f"  Используется процессов: {self.max_workers}")
        print(f"  Сохранять только прибыльные: {self.config.optimization.save_only_profitable}")
        print(f"  Метрика ранжирования: {self.config.optimization.ranking_metric}")
        print()

        # Запускаем параллельную оптимизацию
        return self._execute_parallel_optimization(tasks, progress_callback)

    def _execute_parallel_optimization(self, tasks: List[OptimizationTask],
                                       progress_callback=None) -> List[OptimizationResult]:
        """Выполняет параллельную оптимизацию"""
        self.start_time = time.time()
        self.optimization_results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Отправляем все задачи на выполнение
            future_to_task = {executor.submit(run_single_optimization, task): task
                              for task in tasks}

            # Собираем результаты по мере завершения
            for future in as_completed(future_to_task):
                result = future.result()
                self.optimization_results.append(result)

                if result.success:
                    self.completed_tasks += 1
                else:
                    self.failed_tasks += 1

                # Показываем прогресс
                self._print_progress()

                # Вызываем callback если есть
                if progress_callback:
                    progress_callback(result, self.completed_tasks + self.failed_tasks, self.total_tasks)

        self.end_time = time.time()
        self._print_final_stats()

        return self.optimization_results

    def _generate_optimization_tasks(self, left_range: Tuple[int, int],
                                     right_range: Tuple[int, int]) -> List[OptimizationTask]:
        """Генерирует все задачи оптимизации"""
        tasks = []
        task_id = 0

        config_dict = self.config.to_dict()

        for left_bars in range(left_range[0], left_range[1] + 1):
            for right_bars in range(right_range[0], right_range[1] + 1):
                tasks.append(OptimizationTask(
                    left_bars=left_bars,
                    right_bars=right_bars,
                    task_id=task_id,
                    csv_path=self.config.data.csv_path,
                    config_dict=config_dict
                ))
                task_id += 1

        return tasks

    def _estimate_execution_time(self) -> float:
        """Оценивает время выполнения в минутах"""
        # Предполагаем 4 секунды на задачу и делим на количество процессов
        estimated_seconds = (self.total_tasks * 4) / self.max_workers
        return estimated_seconds / 60

    def _print_progress(self):
        """Выводит прогресс выполнения в виде обновляющейся строки"""
        total_processed = self.completed_tasks + self.failed_tasks
        progress_percent = (total_processed / self.total_tasks) * 100

        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60

        # Создаем прогресс-бар
        bar_width = 40
        filled_width = int(bar_width * progress_percent / 100)
        bar = '█' * filled_width + '░' * (bar_width - filled_width)

        print(f"\r[{bar}] {progress_percent:.1f}% | "
              f"{total_processed:,}/{self.total_tasks:,} | "
              f"✅ {self.completed_tasks:,} | "
              f"❌ {self.failed_tasks} | "
              f"⏱️ {elapsed_minutes:.1f}мин",
              end="", flush=True)

    def _print_final_stats(self):
        """Выводит финальную статистику"""
        print("\n")
        print("=" * 60)
        print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
        print("=" * 60)

        execution_time = self.end_time - self.start_time
        print(f"Время выполнения: {execution_time / 60:.1f} минут")
        print(f"Всего задач: {self.total_tasks:,}")
        print(f"Успешно выполнено: {self.completed_tasks:,}")
        print(f"Ошибок: {self.failed_tasks}")
        print(f"Скорость: {self.total_tasks / execution_time:.1f} задач/сек")

        # Статистика по результатам
        successful_results = [r for r in self.optimization_results if r.success]
        if successful_results:
            profitable_count = len([r for r in successful_results if r.total_return > 0])
            print(f"Прибыльных комбинаций: {profitable_count}/{len(successful_results)} "
                  f"({profitable_count / len(successful_results) * 100:.1f}%)")

            if profitable_count > 0:
                best_result = max(successful_results, key=lambda x: x.total_return)
                print(f"Лучшая комбинация: L={best_result.left_bars}, R={best_result.right_bars} "
                      f"(ROI: {best_result.total_return:.2f}%)")

    def get_top_results(self, top_n: int = None,
                        sort_by: str = None) -> List[OptimizationResult]:
        """
        Возвращает топ-N лучших результатов по указанной метрике
        """
        # Используем настройки из конфигурации если не указаны явно
        if top_n is None:
            top_n = self.config.optimization.top_results_count
        if sort_by is None:
            sort_by = self.config.optimization.ranking_metric

        successful_results = [r for r in self.optimization_results if r.success]

        if not successful_results:
            return []

        # Сортируем по выбранной метрике (по убыванию)
        if sort_by == 'sharpe_ratio':
            # Для Sharpe учитываем только положительные значения
            successful_results = [r for r in successful_results if r.sharpe_ratio > 0]

        try:
            sorted_results = sorted(successful_results,
                                    key=lambda x: getattr(x, sort_by),
                                    reverse=True)
            return sorted_results[:top_n]
        except AttributeError:
            print(f"Ошибка: метрика '{sort_by}' не найдена. Используем 'total_return'")
            sorted_results = sorted(successful_results,
                                    key=lambda x: x.total_return,
                                    reverse=True)
            return sorted_results[:top_n]

    def print_top_results(self, top_n: int = None, sort_by: str = None):
        """Выводит топ результатов в консоль"""
        # Используем настройки из конфигурации если не указаны явно
        if top_n is None:
            top_n = min(20, self.config.optimization.top_results_count)  # Максимум 20 в консоль
        if sort_by is None:
            sort_by = self.config.optimization.ranking_metric

        top_results = self.get_top_results(self.config.optimization.top_results_count, sort_by)[:top_n]

        if not top_results:
            print("Нет успешных результатов для отображения")
            return

        print(f"\nТОП-{len(top_results)} КОМБИНАЦИЙ ПО МЕТРИКЕ '{sort_by.upper()}':")
        print("-" * 120)
        print(
            f"{'#':<3} {'L':<3} {'R':<3} {'ROI%':<8} {'PnL$':<10} {'DD%':<7} {'Sharpe':<7} {'PF':<6} {'WR%':<6} {'Trades':<7} {'Calmar':<7}")
        print("-" * 120)

        for i, result in enumerate(top_results, 1):
            print(f"{i:<3} "
                  f"{result.left_bars:<3} "
                  f"{result.right_bars:<3} "
                  f"{result.total_return:<8.2f} "
                  f"{result.total_pnl:<10.2f} "
                  f"{result.max_drawdown_percent:<7.2f} "
                  f"{result.sharpe_ratio:<7.3f} "
                  f"{result.profit_factor:<6.2f} "
                  f"{result.win_rate:<6.1f} "
                  f"{result.total_trades:<7} "
                  f"{result.calmar_ratio:<7.3f}")

        print("-" * 120)