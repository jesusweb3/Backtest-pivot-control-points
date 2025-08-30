# main.py

import sys
import os
from typing import Optional
from io import StringIO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.backtest_engine import BacktestEngine
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


def main() -> bool:
    """Главная функция запуска упрощенного бэктестера"""

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

        print("-" * 60)
        print("ЗАПУСК БЭКТЕСТА ПО СТРАТЕГИИ КОНТРОЛЬНЫХ ТОЧЕК")
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
            console_capture.stop_capture()
            print("Бэктест завершился с ошибкой")
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

        # Дополняем результаты ключевыми метриками
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
        BacktestVisualizer()
        BacktestVisualizer.create_complete_report(
            equity_curve=results['equity_curve'],
            trades=results['trades'],
            config=results['config']
        )

        print("БЭКТЕСТ УСПЕШНО ЗАВЕРШЕН!")

        # Останавливаем дублирование и сохраняем лог
        console_capture.stop_capture()
        console_output = console_capture.get_output()

        # Сохраняем лог консоли в файл
        recorder.save_console_log(console_output, results['config'])

        return True

    except KeyboardInterrupt:
        console_capture.stop_capture()
        print("\nБэктест прерван пользователем")
        return False
    except Exception as e:
        console_capture.stop_capture()
        print(f"\nОшибка выполнения бэктеста: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test(data_file_path: Optional[str] = None) -> bool:

    config = BacktestConfig()

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
            print("python main.py         - Полный бэктест с отчетами")
            print("python main.py --quick - Быстрый тест без отчетов")
            sys.exit(1)
    else:
        success = main()
        sys.exit(0 if success else 1)