# solo_backtest/recorder.py

import pandas as pd
from typing import List, Dict
from datetime import datetime
import os


class TradeRecorder:
    """
    Записывает детализацию сделок в Excel файлы.
    Создает структуру папок по параметрам стратегии.
    """

    def __init__(self, base_results_path: str = "results"):
        self.base_results_path = base_results_path

    @staticmethod
    def _ensure_directory_exists(path: str) -> None:
        """Создает директорию если она не существует"""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _generate_folder_name(config: Dict) -> str:
        """
        Генерирует имя папки на основе параметров стратегии
        """
        pivot_config = config.get('pivot', {})
        left_bars = pivot_config.get('left_bars', 0)
        right_bars = pivot_config.get('right_bars', 0)

        return f"{left_bars}-{right_bars}"

    def get_results_paths(self, config: Dict) -> Dict[str, str]:

        folder_name = TradeRecorder._generate_folder_name(config)
        base_folder = os.path.join(self.base_results_path, folder_name)

        paths = {
            'base_folder': base_folder,
            'trades_folder': os.path.join(base_folder, 'trades'),
            'charts_folder': os.path.join(base_folder, 'charts'),
            'log_file': os.path.join(base_folder, 'backtest_log.txt')
        }

        for path_key, path_value in paths.items():
            if path_key != 'log_file':
                TradeRecorder._ensure_directory_exists(path_value)
            else:
                TradeRecorder._ensure_directory_exists(os.path.dirname(path_value))

        return paths

    def export_trades_to_excel(self, trades: List[Dict],
                               performance_metrics: Dict,
                               config: Dict,
                               filename: str = None) -> str:
        """
        Экспортирует сделки и метрики в Excel файл в структурированную папку
        """
        paths = self.get_results_paths(config)
        trades_folder = paths['trades_folder']

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.xlsx"

        filepath = os.path.join(trades_folder, filename)

        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Лист с детализацией сделок
                if trades:
                    trades_df = TradeRecorder._prepare_trades_dataframe(trades)
                    if not trades_df.empty:
                        trades_df.to_excel(writer, sheet_name='Trades', index=False)
                        TradeRecorder._format_trades_sheet(writer.sheets['Trades'])

                # Лист с метриками производительности
                metrics_df = TradeRecorder._prepare_metrics_dataframe(performance_metrics)
                metrics_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
                TradeRecorder._format_metrics_sheet(writer.sheets['Performance_Metrics'])

                # Лист с конфигурацией
                config_df = TradeRecorder._prepare_config_dataframe(config)
                config_df.to_excel(writer, sheet_name='Configuration', index=False)
                TradeRecorder._format_config_sheet(writer.sheets['Configuration'])

            print(f"Результаты экспортированы в: {filepath}")
            return filepath

        except Exception as e:
            print(f"Ошибка экспорта в Excel: {e}")
            csv_filepath = filepath.replace('.xlsx', '_trades.csv')
            if trades:
                trades_df = TradeRecorder._prepare_trades_dataframe(trades)
                trades_df.to_csv(csv_filepath, index=False)
                print(f"Сделки экспортированы в CSV: {csv_filepath}")
                return csv_filepath
            return ""

    def save_console_log(self, console_output: str, config: Dict) -> str:
        """
        Сохраняет вывод консоли в текстовый файл
        """
        paths = self.get_results_paths(config)
        log_filepath = paths['log_file']

        try:
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write(console_output)

            print(f"Лог консоли сохранен в: {log_filepath}")
            return log_filepath

        except Exception as e:
            print(f"Ошибка сохранения лога: {e}")
            return ""

    @staticmethod
    def _prepare_trades_dataframe(trades: List[Dict]) -> pd.DataFrame:
        """Подготавливает DataFrame с детализацией сделок на русском языке"""
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])

        if 'entry_time' in df.columns and 'exit_time' in df.columns:
            df['duration_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
            df['duration_hours'] = df['duration_hours'].round(1)

        if 'pnl' in df.columns:
            df['pnl'] = df['pnl'].round(2)
            df['cumulative_pnl'] = df['pnl'].cumsum()
            df['cumulative_pnl'] = df['cumulative_pnl'].round(2)

        direction_column = None
        if 'direction' in df.columns:
            direction_column = 'direction'
            if 'entry_type' in df.columns:
                df = df.drop('entry_type', axis=1)
        elif 'entry_type' in df.columns:
            direction_column = 'entry_type'

        column_mapping = {
            'entry_time': 'Время входа',
            'exit_time': 'Время выхода',
            'duration_hours': 'Продолжительность (в часах)',
            'symbol': 'Актив',
            'quantity': 'Объём',
            'entry_price': 'Цена входа',
            'exit_price': 'Цена выхода',
            'pnl': 'PnL ($)',
            'commission': 'Комиссия ($)',
            'cumulative_pnl': 'Кумулятивный PnL ($)'
        }

        if direction_column:
            column_mapping[direction_column] = 'Направление'

        existing_columns = {old: new for old, new in column_mapping.items() if old in df.columns}
        df = df.rename(columns=existing_columns)

        if 'exit_type' in df.columns:
            df = df.drop('exit_type', axis=1)

        # Упорядочиваем колонки
        desired_order = [
            'Актив', 'Время входа', 'Направление', 'Цена входа', 'Объём',
            'Время выхода', 'Цена выхода', 'Продолжительность (в часах)',
            'PnL ($)', 'Комиссия ($)', 'Кумулятивный PnL ($)'
        ]

        available_columns = [col for col in desired_order if col in df.columns]

        remaining_columns = [col for col in df.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns

        return df[final_columns] if final_columns else df

    @staticmethod
    def _prepare_metrics_dataframe(metrics: Dict) -> pd.DataFrame:
        """Подготавливает DataFrame с метриками производительности на русском языке"""
        metrics_data = []

        # Доходность
        metrics_data.extend([
            ('Доходность', 'Общая доходность (%)', f"{metrics.get('total_return', 0):.2f}"),
            ('Доходность', 'Годовая доходность (%)', f"{metrics.get('annual_return', 0):.2f}"),
            ('Доходность', 'Макс. просадка ($)', f"{metrics.get('max_drawdown', 0):.2f}"),
            ('Доходность', 'Макс. просадка (%)', f"{metrics.get('max_drawdown_percent', 0):.2f}"),
        ])

        # Риск-метрики
        metrics_data.extend([
            ('Риск', 'Коэффициент Шарпа', f"{metrics.get('sharpe_ratio', 0):.3f}"),
            ('Риск', 'Коэффициент Сортино', f"{metrics.get('sortino_ratio', 0):.3f}"),
            ('Риск', 'Волатильность (%)', f"{metrics.get('volatility', 0):.2f}"),
        ])

        # Торговые метрики
        metrics_data.extend([
            ('Торговля', 'Всего сделок', f"{metrics.get('total_trades', 0)}"),
            ('Торговля', 'Прибыльные сделки', f"{metrics.get('winning_trades', 0)}"),
            ('Торговля', 'Убыточные сделки', f"{metrics.get('losing_trades', 0)}"),
            ('Торговля', 'Винрейт (%)', f"{metrics.get('win_rate', 0):.2f}"),
            ('Торговля', 'Профит-фактор', f"{metrics.get('profit_factor', 0):.3f}"),
            ('Торговля', 'Средняя сделка ($)', f"{metrics.get('avg_trade', 0):.2f}"),
            ('Торговля', 'Средний выигрыш ($)', f"{metrics.get('avg_win', 0):.2f}"),
            ('Торговля', 'Средний проигрыш ($)', f"{metrics.get('avg_loss', 0):.2f}"),
            ('Торговля', 'Наибольший выигрыш ($)', f"{metrics.get('largest_win', 0):.2f}"),
            ('Торговля', 'Наибольший проигрыш ($)', f"{metrics.get('largest_loss', 0):.2f}"),
        ])

        # Дополнительные метрики
        metrics_data.extend([
            ('Дополнительно', 'Фактор восстановления', f"{metrics.get('recovery_factor', 0):.3f}"),
            ('Дополнительно', 'Мат. ожидание ($)', f"{metrics.get('expectancy', 0):.2f}"),
            ('Дополнительно', 'Критерий Келли', f"{metrics.get('kelly_criterion', 0):.3f}"),
            ('Дополнительно', 'Ср. продолжительность (ч)', f"{metrics.get('avg_trade_duration', 0):.1f}"),
            ('Дополнительно', 'Период торговли (дн)', f"{metrics.get('trading_period_days', 0)}"),
        ])

        df = pd.DataFrame(metrics_data, columns=['Категория', 'Метрика', 'Значение'])
        return df

    @staticmethod
    def _prepare_config_dataframe(config: Dict) -> pd.DataFrame:
        """Подготавливает DataFrame с конфигурацией на русском языке"""
        config_data = []

        # Trading конфигурация
        trading_config = config.get('trading', {})
        trading_translations = {
            'initial_capital': 'Начальный капитал',
            'leverage': 'Плечо',
            'position_size': 'Размер позиции',
            'taker_commission': 'Комиссия тейкера',
            'min_tick': 'Минимальный тик'
        }

        for key, value in trading_config.items():
            translated_key = trading_translations.get(key, key.replace('_', ' ').title())
            config_data.append(('Торговля', translated_key, str(value)))

        # Pivot конфигурация
        pivot_config = config.get('pivot', {})
        pivot_translations = {
            'left_bars': 'Левые бары',
            'right_bars': 'Правые бары'
        }

        for key, value in pivot_config.items():
            translated_key = pivot_translations.get(key, key.replace('_', ' ').title())
            config_data.append(('Стратегия', translated_key, str(value)))

        # Data конфигурация
        data_config = config.get('data', {})
        data_translations = {
            'symbol': 'Символ',
            'csv_path': 'Путь к данным',
            'timeframe': 'Таймфрейм'
        }

        for key, value in data_config.items():
            translated_key = data_translations.get(key, key.replace('_', ' ').title())
            config_data.append(('Данные', translated_key, str(value)))

        df = pd.DataFrame(config_data, columns=['Категория', 'Параметр', 'Значение'])
        return df

    @staticmethod
    def _format_trades_sheet(worksheet) -> None:
        """Форматирует лист с сделками: автоширина, закрепление заголовка, выравнивание по центру"""
        from openpyxl.styles import Alignment

        # Закрепляем первую строку (заголовки)
        worksheet.freeze_panes = 'A2'

        # Автоматическая настройка ширины колонок под содержимое
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    # Получаем длину содержимого ячейки
                    cell_length = len(str(cell.value)) if cell.value else 0
                    if cell_length > max_length:
                        max_length = cell_length
                except (AttributeError, TypeError):
                    pass

            # Устанавливаем ширину с небольшим запасом, но не более 50 символов
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Выравнивание по центру для всех ячеек
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(horizontal='center', vertical='center')

    @staticmethod
    def _format_metrics_sheet(worksheet) -> None:
        """Форматирует лист с метриками"""
        worksheet.column_dimensions['A'].width = 15
        worksheet.column_dimensions['B'].width = 25
        worksheet.column_dimensions['C'].width = 15

    @staticmethod
    def _format_config_sheet(worksheet) -> None:
        """Форматирует лист с конфигурацией"""
        worksheet.column_dimensions['A'].width = 15
        worksheet.column_dimensions['B'].width = 25
        worksheet.column_dimensions['C'].width = 20