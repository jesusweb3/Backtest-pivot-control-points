# main.py

import sys
import os
from typing import Optional
from io import StringIO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.backtest_engine import BacktestEngine
from core.parameter_optimizer import ParameterOptimizer
from analytics.performance_analyzer import PerformanceAnalyzer
from analytics.trade_recorder import TradeRecorder
from analytics.visualizer import BacktestVisualizer
from config.settings import BacktestConfig


class ConsoleCapture:
    """Класс для дублирования вывода консоли в строку и на экран одновременно"""

    def __init__(self):
        self.captured_output = StringIO()
        self.original_stdout = sys.stdout

    def start_capture(self):
        """Начинает дублирование вывода"""
        sys.stdout = self

    def stop_capture(self):
        """Останавливает дублирование и возвращает оригинальный stdout"""
        sys.stdout = self.original_stdout

    def write(self, text):
        """Записывает текст и в лог, и на экран"""
        self.captured_output.write(text)
        self.original_stdout.write(text)

    def flush(self):
        """Очищает буферы"""
        self.captured_output.flush()
        self.original_stdout.flush()

    def get_output(self) -> str:
        """Возвращает перехваченный вывод"""
        return self.captured_output.getvalue()


def run_single_backtest(config: BacktestConfig) -> bool:
    """Запускает обычный единичный бэктест"""

    print("-" * 60)
    print("ЗАПУСК ЕДИНИЧНОГО БЭКТЕСТА")
    print("-" * 60)
    print(f"Конфигурация:")
    print(f"  Символ: {config.data.symbol}")
    print(f"  Начальный капитал: ${config.trading.initial_capital:,.2f}")
    print(f"  Размер маржи: ${config.trading.position_size:,.2f}")
    print(f"  Плечо: {config.trading.leverage}x")
    print(f"  Комиссия: {config.trading.taker_commission * 100:.3f}%")
    print(f"  Pivot параметры: L={config.pivot.left_bars}, R={config.pivot.right_bars}")
    print()

    # Инициализируем и запускаем движок
    engine = BacktestEngine(config)
    backtest_success = engine.run_backtest()

    if not backtest_success:
        return False

    # Получаем результаты
    results = engine.get_results()

    # Анализируем производительность
    analyzer = PerformanceAnalyzer(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        initial_capital=config.trading.initial_capital
    )

    performance_metrics = analyzer.calculate_all_metrics()

    # Выводим дополнительные метрики
    winning_trades = len([t for t in results['trades'] if t['pnl'] > 0])
    losing_trades = len([t for t in results['trades'] if t['pnl'] <= 0])
    total_commission = sum(t['commission'] for t in results['trades'])

    print(f"Максимальная просадка: {performance_metrics['max_drawdown_percent']:.2f}%")
    print(f"Винрейт: {winning_trades / max(1, len(results['trades'])) * 100:.1f}%")
    print(f"Всего сделок: {len(results['trades'])}")
    print(f"Прибыльные: {winning_trades}")
    print(f"Убыточные: {losing_trades}")
    print(f"Общие комиссии: ${total_commission:.2f}")
    print()
    print(f"Коэффициент Шарпа: {performance_metrics['sharpe_ratio']:.3f}")
    print(f"Профит-фактор: {performance_metrics['profit_factor']:.3f}")
    print(f"Средняя сделка: ${performance_metrics['avg_trade']:.2f}")

    # Экспорт результатов
    print("-" * 60)
    print("ЭКСПОРТ РЕЗУЛЬТАТОВ")
    print("-" * 60)

    recorder = TradeRecorder()
    recorder.export_trades_to_excel(
        trades=results['trades'],
        performance_metrics=performance_metrics,
        config=results['config']
    )

    print("Создание графиков...")
    BacktestVisualizer.create_complete_report(
        equity_curve=results['equity_curve'],
        trades=results['trades'],
        config=results['config']
    )

    return True


def run_parameter_optimization(config: BacktestConfig) -> bool:
    """Запускает параметрическую оптимизацию"""

    # Инициализируем оптимизатор
    optimizer = ParameterOptimizer(config)

    try:
        # Запускаем оптимизацию
        optimization_results = optimizer.run_optimization()

        if not optimization_results:
            print("Оптимизация не дала результатов")
            return False

        # Показываем топ результатов в консоли
        print()
        optimizer.print_top_results(top_n=20)

        # Экспорт топ результатов
        print()
        print("-" * 60)
        print("ЭКСПОРТ РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
        print("-" * 60)

        top_results = optimizer.get_top_results()

        if top_results:
            # Создаем сводный отчет по оптимизации
            export_optimization_summary(top_results, config)

            # Создаем графики для ВСЕХ топ результатов (топ-1000)
            print("Создание графиков для всех топ результатов...")
            create_optimization_charts(top_results, config)

            print(f"Экспорт завершен. Топ-{len(top_results)} результатов сохранены.")
            print(f"Метрика ранжирования: {config.optimization.ranking_metric}")

        return True

    except Exception as e:
        print(f"Ошибка оптимизации: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_optimization_summary(results, config: BacktestConfig):
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
                ('Всего комбинаций', config.get_total_combinations()),
                ('Успешных результатов', len([r for r in results if r.success])),
                ('Прибыльных комбинаций', len([r for r in results if r.total_return > 0])),
                ('Диапазон Left Bars',
                 f"{config.optimization.left_bars_range[0]}-{config.optimization.left_bars_range[1]}"),
                ('Диапазон Right Bars',
                 f"{config.optimization.right_bars_range[0]}-{config.optimization.right_bars_range[1]}"),
                ('Метрика ранжирования', config.optimization.ranking_metric),
                ('Максимум процессов', config.optimization.max_workers or 'Авто'),
                ('Сохранять только прибыльные', config.optimization.save_only_profitable)
            ]

            stats_df = pd.DataFrame(stats_data[1:], columns=stats_data[0])
            stats_df.to_excel(writer, sheet_name='Optimization_Stats', index=False)

        print(f"Сводка оптимизации сохранена: {filepath}")

    except Exception as e:
        print(f"Ошибка экспорта сводки: {e}")


def create_optimization_charts(top_results, config: BacktestConfig):
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
                executor.submit(create_single_chart, result, config.to_dict()): result
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


def create_single_chart(result, config_dict):
    """
    Создает графики для одного результата оптимизации
    ВАЖНО: функция должна быть на верхнем уровне модуля для pickle
    """
    import io
    import sys
    old_stdout = None

    try:
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
            config=results_data['config'],  # Используем конфигурацию результата
            filename=f"optimization_analysis.png"
        )

        # Создаем график месячной доходности если данных достаточно
        if len(results_data['equity_curve']) > 30:
            BacktestVisualizer.create_monthly_returns_chart(
                equity_curve=results_data['equity_curve'],
                config=results_data['config'],
                filename=f"optimization_monthly.png"
            )

        return True

    except (ValueError, AttributeError, TypeError, ImportError, OSError):
        return False
    finally:
        # Восстанавливаем вывод в любом случае
        if old_stdout is not None:
            sys.stdout = old_stdout


def main() -> bool:
    """Главная функция - выбирает режим работы по конфигурации"""

    # Инициализируем перехват консоли
    console_capture = ConsoleCapture()

    # Загружаем конфигурацию
    config = BacktestConfig()

    # Проверяем наличие файла данных
    if not os.path.exists(config.data.csv_path):
        print(f"Ошибка: Файл данных не найден: {config.data.csv_path}")
        print("Создайте папку 'data' и поместите туда CSV файл с данными")
        print("Формат CSV: timestamp,open,high,low,close")
        return False

    try:
        # Начинаем дублирование консольного вывода
        console_capture.start_capture()

        # Выбираем режим работы
        if config.is_optimization_mode():
            print("🔥 РЕЖИМ ПАРАМЕТРИЧЕСКОЙ ОПТИМИЗАЦИИ")
            execution_success = run_parameter_optimization(config)
        else:
            print("📊 РЕЖИМ ЕДИНИЧНОГО БЭКТЕСТА")
            execution_success = run_single_backtest(config)

        if execution_success:
            print("\n" + "=" * 60)
            print("ВЫПОЛНЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            print("=" * 60)

        # Останавливаем дублирование и сохраняем лог
        console_capture.stop_capture()
        console_output = console_capture.get_output()

        # Сохраняем лог консоли в файл только для единичного бэктеста
        try:
            if not config.is_optimization_mode():
                # Только для единичного бэктеста сохраняем лог
                recorder = TradeRecorder()
                config_dict = config.to_dict()
                config_dict['data']['csv_path'] = config.data.csv_path
                recorder.save_console_log(console_output, config_dict)
        except Exception as e:
            print(f"Не удалось сохранить лог: {e}")

        return execution_success

    except KeyboardInterrupt:
        console_capture.stop_capture()
        print("\nВыполнение прервано пользователем")
        return False
    except Exception as e:
        console_capture.stop_capture()
        print(f"\nОшибка выполнения: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test(data_file_path: Optional[str] = None) -> bool:
    """Быстрый тест (всегда единичный бэктест)"""

    config = BacktestConfig()
    config.optimization.enable_optimization = False  # Принудительно отключаем оптимизацию

    if data_file_path:
        config.data.csv_path = data_file_path

    print("🚀 БЫСТРЫЙ ТЕСТ")
    print("-" * 30)

    engine = BacktestEngine(config)
    test_success = engine.run_backtest()

    if test_success:
        results = engine.get_results()
        print(f"\n📊 Быстрая сводка:")
        print(f"   Сделок: {len(results['trades'])}")
        print(f"   PnL: ${results['total_pnl']:+.2f}")
        print(f"   ROI: {(results['total_pnl'] / config.trading.initial_capital) * 100:+.1f}%")

    return test_success


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            data_path = sys.argv[2] if len(sys.argv) > 2 else None
            test_result = quick_test(data_path)
            sys.exit(0 if test_result else 1)
        else:
            print(f"Неизвестный параметр: {sys.argv[1]}")
            print("\nДоступные команды:")
            print("python main.py         - Запуск согласно конфигурации (бэктест или оптимизация)")
            print("python main.py --quick - Быстрый единичный тест без отчетов")
            print("\nРежим определяется параметром config.optimization.enable_optimization:")
            print("  False = Единичный бэктест")
            print("  True  = Параметрическая оптимизация")
            sys.exit(1)
    else:
        success = main()
        sys.exit(0 if success else 1)