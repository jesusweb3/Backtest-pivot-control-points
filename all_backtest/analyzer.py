# all_backtest/analyzer.py

import numpy as np
import pandas as pd
from typing import List, Dict, Any


class PerformanceAnalyzer:
    """
    Анализатор производительности торговой стратегии.
    Рассчитывает все основные метрики для оценки эффективности.
    """

    def __init__(self, equity_curve: List[Dict], trades: List[Dict],
                 initial_capital: float):
        self.equity_curve = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        self.equity_df = pd.DataFrame(equity_curve)
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        if not self.equity_df.empty:
            self.equity_df['timestamp'] = pd.to_datetime(self.equity_df['timestamp'])
            self.equity_df = self.equity_df.sort_values('timestamp')

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Рассчитывает все метрики производительности"""
        if self.equity_df.empty:
            return PerformanceAnalyzer._empty_metrics()

        metrics = {
            # Основные метрики доходности
            'total_return': self._calculate_total_return(),
            'annual_return': self._calculate_annual_return(),
            'max_drawdown': self._calculate_max_drawdown(),
            'max_drawdown_percent': self._calculate_max_drawdown_percent(),

            # Риск-метрики
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'sortino_ratio': self._calculate_sortino_ratio(),
            'volatility': self._calculate_volatility(),

            # Торговые метрики
            'total_trades': len(self.trades),
            'winning_trades': self._count_winning_trades(),
            'losing_trades': self._count_losing_trades(),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'avg_trade': self._calculate_average_trade(),
            'avg_win': self._calculate_average_win(),
            'avg_loss': self._calculate_average_loss(),
            'largest_win': self._calculate_largest_win(),
            'largest_loss': self._calculate_largest_loss(),

            # Временные метрики
            'avg_trade_duration': self._calculate_avg_trade_duration(),
            'trading_period_days': self._calculate_trading_period(),

            # Дополнительные метрики
            'recovery_factor': self._calculate_recovery_factor(),
            'expectancy': self._calculate_expectancy(),
            'kelly_criterion': self._calculate_kelly_criterion()
        }

        return metrics

    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        """Возвращает пустые метрики если нет данных"""
        return {key: 0.0 for key in [
            'total_return', 'annual_return', 'max_drawdown', 'max_drawdown_percent',
            'sharpe_ratio', 'sortino_ratio', 'volatility',
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'profit_factor', 'avg_trade', 'avg_win', 'avg_loss',
            'largest_win', 'largest_loss', 'avg_trade_duration',
            'trading_period_days', 'recovery_factor', 'expectancy', 'kelly_criterion'
        ]}

    def _calculate_total_return(self) -> float:
        """Общая доходность в процентах"""
        if self.equity_df.empty:
            return 0.0
        final_equity = self.equity_df['equity'].iloc[-1]
        return (final_equity / self.initial_capital - 1) * 100

    def _calculate_annual_return(self) -> float:
        """Годовая доходность с защитой от математических ошибок"""
        if self.equity_df.empty or len(self.equity_df) < 2:
            return 0.0

        days = self._calculate_trading_period()
        if days == 0:
            return 0.0

        total_return_decimal = self._calculate_total_return() / 100

        base_value = 1 + total_return_decimal

        if base_value <= 0:
            return -99.99

        try:
            annual_return = (base_value ** (365.25 / days)) - 1
            return annual_return * 100
        except (ValueError, OverflowError, ZeroDivisionError):
            return -99.99 if total_return_decimal < 0 else 0.0

    def _calculate_max_drawdown(self) -> float:
        """Максимальная просадка в абсолютном выражении"""
        if self.equity_df.empty:
            return 0.0

        equity = self.equity_df['equity']
        peak = equity.expanding().max()
        drawdown = peak - equity
        return drawdown.max()

    def _calculate_max_drawdown_percent(self) -> float:
        """Максимальная просадка в процентах"""
        if self.equity_df.empty:
            return 0.0

        equity = self.equity_df['equity']
        peak = equity.expanding().max()
        drawdown_pct = (peak - equity) / peak * 100
        return drawdown_pct.max()

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Коэффициент Шарпа с улучшенной логикой"""
        if self.equity_df.empty or len(self.equity_df) < 3:  # Минимум 3 точки
            return 0.0

        returns = self.equity_df['equity'].pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Рассчитываем периодичность (дневная безрисковая ставка)
        daily_rf_rate = risk_free_rate / 365.25
        excess_returns = returns - daily_rf_rate

        sharpe = excess_returns.mean() / returns.std()

        # Аннуализируем (предполагаем что returns - это изменения по сделкам)
        # Используем sqrt от количества сделок в году (приблизительно)
        annualization_factor = np.sqrt(252) if len(returns) > 1 else 1.0
        sharpe_annualized = sharpe * annualization_factor

        # Защита от NaN и бесконечности
        if np.isnan(sharpe_annualized) or np.isinf(sharpe_annualized):
            return 0.0

        return sharpe_annualized

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Коэффициент Сортино"""
        if self.equity_df.empty or len(self.equity_df) < 2:
            return 0.0

        returns = self.equity_df['equity'].pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252

        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_calmar_ratio(self) -> float:
        """Коэффициент Калмара"""
        annual_return = self._calculate_annual_return()
        max_dd_pct = self._calculate_max_drawdown_percent()

        if max_dd_pct == 0:
            return 0.0
        return annual_return / max_dd_pct

    def _calculate_volatility(self) -> float:
        """Годовая волатильность"""
        if self.equity_df.empty or len(self.equity_df) < 2:
            return 0.0

        returns = self.equity_df['equity'].pct_change().dropna()
        return returns.std() * np.sqrt(252) * 100

    def _count_winning_trades(self) -> int:
        """Количество прибыльных сделок"""
        if self.trades_df.empty:
            return 0
        return len(self.trades_df[self.trades_df['pnl'] > 0])

    def _count_losing_trades(self) -> int:
        """Количество убыточных сделок"""
        if self.trades_df.empty:
            return 0
        return len(self.trades_df[self.trades_df['pnl'] <= 0])

    def _calculate_win_rate(self) -> float:
        """Винрейт в процентах"""
        if self.trades_df.empty:
            return 0.0
        return self._count_winning_trades() / len(self.trades_df) * 100

    def _calculate_profit_factor(self) -> float:
        """Профит-фактор"""
        if self.trades_df.empty:
            return 0.0

        gross_profit = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.trades_df[self.trades_df['pnl'] <= 0]['pnl'].sum())

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _calculate_average_trade(self) -> float:
        """Средняя прибыль на сделку"""
        if self.trades_df.empty:
            return 0.0
        return self.trades_df['pnl'].mean()

    def _calculate_average_win(self) -> float:
        """Средняя прибыль на прибыльную сделку"""
        if self.trades_df.empty:
            return 0.0
        winning_trades = self.trades_df[self.trades_df['pnl'] > 0]['pnl']
        return winning_trades.mean() if not winning_trades.empty else 0.0

    def _calculate_average_loss(self) -> float:
        """Средний убыток на убыточную сделку"""
        if self.trades_df.empty:
            return 0.0
        losing_trades = self.trades_df[self.trades_df['pnl'] <= 0]['pnl']
        return losing_trades.mean() if not losing_trades.empty else 0.0

    def _calculate_largest_win(self) -> float:
        """Наибольшая прибыль"""
        if self.trades_df.empty:
            return 0.0
        return self.trades_df['pnl'].max()

    def _calculate_largest_loss(self) -> float:
        """Наибольший убыток"""
        if self.trades_df.empty:
            return 0.0
        return self.trades_df['pnl'].min()

    def _calculate_avg_trade_duration(self) -> float:
        """Средняя продолжительность сделки в часах"""
        if self.trades_df.empty or 'entry_time' not in self.trades_df.columns:
            return 0.0

        self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
        self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])

        durations = (self.trades_df['exit_time'] - self.trades_df['entry_time']).dt.total_seconds() / 3600
        return durations.mean()

    def _calculate_trading_period(self) -> float:
        """Общий период торговли в днях"""
        if self.equity_df.empty:
            return 0.0

        start_date = self.equity_df['timestamp'].iloc[0]
        end_date = self.equity_df['timestamp'].iloc[-1]
        return (end_date - start_date).days

    def _calculate_recovery_factor(self) -> float:
        """Фактор восстановления"""
        total_return = self._calculate_total_return()
        max_dd_pct = self._calculate_max_drawdown_percent()

        if max_dd_pct == 0:
            return 0.0
        return total_return / max_dd_pct

    def _calculate_expectancy(self) -> float:
        """Математическое ожидание"""
        if self.trades_df.empty:
            return 0.0

        win_rate = self._calculate_win_rate() / 100
        avg_win = self._calculate_average_win()
        avg_loss = abs(self._calculate_average_loss())

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def _calculate_kelly_criterion(self) -> float:
        """Критерий Келли"""
        if self.trades_df.empty:
            return 0.0

        win_rate = self._calculate_win_rate() / 100
        avg_win = self._calculate_average_win()
        avg_loss = abs(self._calculate_average_loss())

        if avg_loss == 0:
            return 0.0

        return win_rate - ((1 - win_rate) / (avg_win / avg_loss))