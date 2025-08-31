# backtest/console_manager.py

import sys
from io import StringIO
from typing import Optional


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

    def clear_buffer(self):
        """Очищает буфер лога"""
        self.log_buffer = StringIO()
        if self.tee_output:
            self.tee_output.log_file = self.log_buffer


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