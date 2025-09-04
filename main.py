# main.py

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from settings.settings import BacktestConfig


def main() -> bool:
    """Главная функция - выбирает режим работы по конфигурации"""

    # Загружаем пользовательскую конфигурацию
    config = BacktestConfig.load_from_user_config()

    # Проверяем наличие файла данных
    if not os.path.exists(config.data.csv_path):
        print(f"Ошибка: Файл данных не найден: {config.data.csv_path}")
        print("Создайте папку 'data' и поместите туда CSV файл с данными")
        print("Формат CSV: timestamp,open,high,low,close")
        return False

    # Выбираем режим работы по конфигурации
    if config.is_optimization_mode():
        print("🔥 РЕЖИМ ПАРАМЕТРИЧЕСКОЙ ОПТИМИЗАЦИИ")
        from all_backtest.runner import AllBacktestRunner
        runner = AllBacktestRunner(config)
    else:
        print("📊 РЕЖИМ ЕДИНИЧНОГО БЭКТЕСТА")
        from solo_backtest.runner import SoloBacktestRunner
        runner = SoloBacktestRunner(config)

    # Запускаем выбранный режим
    return runner.run()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)