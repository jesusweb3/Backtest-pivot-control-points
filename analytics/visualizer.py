# analytics/visualizer.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict
import os
from datetime import datetime


class BacktestVisualizer:
    """
    Создает графики и визуализации для анализа результатов бэктеста.
    Использует структуру папок по параметрам стратегии.
    """

    def __init__(self):
        BacktestVisualizer._setup_matplotlib_style()

    @staticmethod
    def _setup_matplotlib_style() -> None:
        """Настраивает темную тему matplotlib"""
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.2
        plt.rcParams['grid.color'] = '#444444'
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = '#666666'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['figure.facecolor'] = '#1a1a1a'
        plt.rcParams['axes.facecolor'] = '#2d2d2d'

    @staticmethod
    def _get_charts_folder(config: Dict) -> str:

        from analytics.trade_recorder import TradeRecorder
        recorder = TradeRecorder()
        paths = recorder.get_results_paths(config)
        return paths['charts_folder']

    @staticmethod
    def create_monthly_returns_chart(equity_curve: List[Dict],
                                     config: Dict,
                                     filename: str = None) -> str:

        if not equity_curve:
            return ""

        charts_folder = BacktestVisualizer._get_charts_folder(config)

        df = pd.DataFrame(equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Ресэмплируем по месяцам
        monthly_equity = df['equity'].resample('ME').last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100

        if monthly_returns.empty:
            return ""

        fig, ax = plt.subplots(figsize=(14, 8))

        colors = ['#00ff7f' if x >= 0 else '#ff6b6b' for x in monthly_returns]

        bars = ax.bar(range(len(monthly_returns)), monthly_returns.values,
                      color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

        ax.set_title('Месячная доходность', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Месяц', fontsize=12)
        ax.set_ylabel('Доходность (%)', fontsize=12)
        ax.grid(True, alpha=0.2, axis='y')
        ax.axhline(y=0, color='white', linewidth=1, alpha=0.7)

        # Подписи на оси X
        month_labels = [d.strftime('%Y-%m') for d in monthly_returns.index]
        ax.set_xticks(range(len(monthly_returns)))
        ax.set_xticklabels(month_labels, rotation=45, ha='right')

        # Добавляем значения на столбцы
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + (0.1 if height >= 0 else -0.5),
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9)

        plt.tight_layout()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monthly_returns_{timestamp}.png"

        filepath = os.path.join(charts_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"График месячной доходности сохранен: {filepath}")
        return filepath

    @staticmethod
    def create_trade_analysis_chart(trades: List[Dict],
                                    config: Dict,
                                    filename: str = None) -> str:

        if not trades:
            return ""

        charts_folder = BacktestVisualizer._get_charts_folder(config)
        df = pd.DataFrame(trades)

        fig, ((ax_main, ax_metrics), (ax_ratio, ax_direction)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#1a1a1a')

        # ============= ОСНОВНОЙ ГРАФИК: Cumulative PnL =============
        cumulative_pnl = df['pnl'].cumsum().values
        trade_numbers = np.array(range(1, len(cumulative_pnl) + 1))

        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl

        ax_main.plot(trade_numbers, cumulative_pnl, color='#00ff7f', linewidth=3)

        max_drawdown_idx = np.argmax(drawdown)
        max_drawdown_value = drawdown[max_drawdown_idx]
        max_drawdown_pnl = cumulative_pnl[max_drawdown_idx]
        ax_main.axhline(y=max_drawdown_pnl, color='#ff6b6b', linestyle='--', linewidth=2,
                        alpha=0.8, label=f'Макс. просадка: ${max_drawdown_value:.2f}')

        ax_main.scatter(trade_numbers[-1], cumulative_pnl[-1],
                        color='#ffd700', s=120, marker='o', zorder=5, edgecolors='white', linewidth=2)

        ax_main.set_title('Кумулятивный PnL', fontsize=14, fontweight='bold', color='white')
        ax_main.set_xlabel('Номер сделки', fontsize=11, color='white')
        ax_main.set_ylabel('PnL ($)', fontsize=11, color='white')
        ax_main.grid(True, alpha=0.2, color='#444444')
        ax_main.axhline(y=0, color='white', linestyle='-', alpha=0.5, linewidth=1)
        ax_main.set_facecolor('#2d2d2d')

        # ============= БЛОК МЕТРИК (правый верхний) =============
        final_pnl = cumulative_pnl[-1]
        max_pnl = np.max(cumulative_pnl)
        win_streak = BacktestVisualizer._calculate_max_win_streak(df['pnl'].values)
        loss_streak = BacktestVisualizer._calculate_max_loss_streak(df['pnl'].values)

        # Дополнительные метрики
        winning_trades = len(df[df['pnl'] > 0])
        total_trades = len(df)
        win_rate = (winning_trades / total_trades) * 100
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0

        # Убираем оси и создаем стильный темный блок
        ax_metrics.axis('off')
        ax_metrics.set_facecolor('#2d2d2d')

        metrics_text = f'''ПОКАЗАТЕЛИ ЭФФЕКТИВНОСТИ

Итоговый PnL: ${final_pnl:.2f}
Максимальный PnL: ${max_pnl:.2f}
Максимальная просадка: ${max_drawdown_value:.2f}

Винрейт: {win_rate:.1f}%
Средняя прибыль: ${avg_win:.2f}
Средний убыток: ${avg_loss:.2f}

Макс. серия побед: {win_streak}
Макс. серия поражений: {loss_streak}

Всего сделок: {total_trades}'''

        ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes,
                        verticalalignment='top', horizontalalignment='left', fontsize=12,
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.8', facecolor='#404040',
                                  edgecolor='#666666', alpha=0.9))

        # ============= Win/Loss Ratio (левый нижний) =============
        losing_trades = total_trades - winning_trades
        labels = ['Прибыльные', 'Убыточные']
        sizes = [winning_trades, losing_trades]
        # Более яркие цвета для темной темы
        colors = ['#00ff7f', '#ff6b6b']

        wedges, texts, autotexts = ax_ratio.pie(sizes, labels=labels, colors=colors,
                                                autopct='%1.1f%%', startangle=90,
                                                textprops={'fontsize': 10, 'color': 'white'},
                                                wedgeprops=dict(edgecolor='white', linewidth=2))

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)

        for text in texts:
            text.set_color('white')
            text.set_fontweight('bold')

        ax_ratio.set_title('Соотношение прибыль/убыток', fontsize=12, fontweight='bold', color='white')
        ax_ratio.set_facecolor('#2d2d2d')

        # ============= PnL by Direction (правый нижний) =============
        if 'direction' in df.columns:
            direction_pnl = df.groupby('direction')['pnl'].sum()
        elif 'entry_type' in df.columns:
            direction_pnl = df.groupby('entry_type')['pnl'].sum()
        else:
            direction_pnl = pd.Series()

        if not direction_pnl.empty:
            bars = ax_direction.bar(direction_pnl.index, direction_pnl.values,
                                    color=['#00ff7f' if x >= 0 else '#ff6b6b' for x in direction_pnl.values],
                                    alpha=0.8, edgecolor='white', linewidth=1)
            ax_direction.set_title('PnL по направлениям', fontsize=12, fontweight='bold', color='white')
            ax_direction.set_ylabel('Общий PnL ($)', fontsize=10, color='white')
            ax_direction.grid(True, alpha=0.2, axis='y', color='#444444')
            ax_direction.tick_params(colors='white')
            ax_direction.set_facecolor('#2d2d2d')

            for bar in bars:
                height = bar.get_height()
                y_offset = abs(height) * 0.05 if height >= 0 else -abs(height) * 0.05
                ax_direction.text(bar.get_x() + bar.get_width() / 2.,
                                  height + y_offset,
                                  f'${height:.0f}', ha='center',
                                  va='bottom' if height >= 0 else 'top', fontsize=10,
                                  color='white', fontweight='bold')

            y_max = max(direction_pnl.values) if direction_pnl.values.max() > 0 else 0
            if y_max > 0:
                ax_direction.set_ylim(top=y_max * 1.15)
        else:
            ax_direction.text(0.5, 0.5, 'Нет данных по направлениям', ha='center', va='center',
                              transform=ax_direction.transAxes, fontsize=12, color='white')
            ax_direction.set_title('PnL по направлениям', fontsize=12, fontweight='bold', color='white')
            ax_direction.set_facecolor('#2d2d2d')

        plt.suptitle('Панель анализа сделок', fontsize=16, fontweight='bold', color='white', y=0.95)
        plt.tight_layout(rect=(0, 0.02, 1, 0.92))

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_analysis_{timestamp}.png"

        filepath = os.path.join(charts_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"График анализа сделок сохранен: {filepath}")
        return filepath

    @staticmethod
    def _calculate_max_win_streak(pnl_values: np.ndarray) -> int:
        """Рассчитывает максимальную серию прибыльных сделок"""
        max_streak = 0
        current_streak = 0

        for pnl in pnl_values:
            if pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def _calculate_max_loss_streak(pnl_values: np.ndarray) -> int:
        """Рассчитывает максимальную серию убыточных сделок"""
        max_streak = 0
        current_streak = 0

        for pnl in pnl_values:
            if pnl <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def create_complete_report(equity_curve: List[Dict], trades: List[Dict],
                               config: Dict,
                               filename_prefix: str = None) -> List[str]:

        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"backtest_{timestamp}"

        created_files = []

        # Месячная доходность (создаем только если данных достаточно)
        if len(equity_curve) > 30:
            monthly_file = BacktestVisualizer.create_monthly_returns_chart(
                equity_curve, config, f"{filename_prefix}_monthly.png"
            )
            if monthly_file:
                created_files.append(monthly_file)

        # Анализ сделок (квадратная компоновка 2x2)
        trades_file = BacktestVisualizer.create_trade_analysis_chart(
            trades, config, f"{filename_prefix}_analysis.png"
        )
        if trades_file:
            created_files.append(trades_file)

        print(f"Создано графиков: {len(created_files)}")
        return created_files