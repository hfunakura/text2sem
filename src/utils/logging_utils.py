import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import threading

import colorama
from colorama import Fore, Style

colorama.init()


class LoggingError(Exception):
    """Logging related errors"""
    pass


class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = (
                self.COLORS[record.levelname] + record.levelname + Style.RESET_ALL
            )

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


class StructuredLogger:
    """Structured logger for metrics and events"""

    def __init__(self, name: str, log_file: Union[str, Path]):
        self.name = name
        self.log_file = Path(log_file)
        self.logger = self._setup_logger()
        self._lock = threading.Lock()

    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger with JSON formatter"""
        logger = logging.getLogger(f"{self.name}_structured")
        logger.setLevel(logging.INFO)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setFormatter(JSONFormatter())
            logger.addHandler(file_handler)
        except PermissionError:
            raise LoggingError("Permission denied: Cannot create log file")
        except Exception as e:
            raise LoggingError(f"Error setting up structured logger: {e}")

        logger.propagate = False
        return logger

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        """Log metrics in structured format"""
        with self._lock:
            log_data = {"type": "metrics"}

            if prefix:
                for key, value in metrics.items():
                    log_data[f"{prefix}_{key}"] = value
            else:
                log_data.update(metrics)

            record = self.logger.makeRecord(
                self.logger.name, logging.INFO, "", 0, "metrics", (), None
            )

            for key, value in log_data.items():
                setattr(record, key, value)

            self.logger.handle(record)

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log structured event"""
        with self._lock:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "event",
                "event_type": event_type,
            }
            log_data.update(data)

            record = self.logger.makeRecord(
                self.logger.name, logging.INFO, "", 0, f"event: {event_type}", (), None
            )

            for key, value in log_data.items():
                setattr(record, key, value)

            self.logger.handle(record)


_loggers: Dict[str, logging.Logger] = {}
_logger_lock = threading.Lock()


def setup_logger(
    name: str,
    log_file: Union[str, Path],
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 10,
) -> logging.Logger:
    """Setup logger with file and console handlers"""

    with _logger_lock:
        if name in _loggers:
            return _loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            )

        try:
            logging.Formatter(format_string)
        except (ValueError, KeyError) as e:
            raise LoggingError(f"Invalid log format: {e}")

        log_file = Path(log_file)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(file_handler)

        except PermissionError:
            raise LoggingError("Permission denied: Cannot create log file")
        except OSError as e:
            if "No space left on device" in str(e):
                raise LoggingError("Disk space: No space left on device")
            raise LoggingError(f"Error creating log file: {e}")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(format_string))
        logger.addHandler(console_handler)

        logger.propagate = False
        _loggers[name] = logger

        return logger


def get_logger(
    name: str, log_file: Optional[Union[str, Path]] = None, **kwargs
) -> logging.Logger:
    """Get or create logger (singleton pattern)"""
    if name in _loggers:
        return _loggers[name]

    with _logger_lock:
        if name in _loggers:
            return _loggers[name]

        if log_file is None:
            log_file = Path("logs") / f"{name}.log"

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )

        try:
            logging.Formatter(format_string)
        except (ValueError, KeyError) as e:
            raise LoggingError(f"Invalid log format: {e}")

        log_file = Path(log_file)
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=10,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(file_handler)

        except PermissionError:
            raise LoggingError("Permission denied: Cannot create log file")
        except OSError as e:
            if "No space left on device" in str(e):
                raise LoggingError("Disk space: No space left on device")
            raise LoggingError(f"Error creating log file: {e}")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredFormatter(format_string))
        logger.addHandler(console_handler)

        logger.propagate = False
        _loggers[name] = logger

        return logger


def log_metrics(metrics: Dict[str, Any], logger_name: str = "metrics") -> None:
    """Convenience function for logging metrics"""
    logger = get_logger(logger_name)

    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    logger.info(f"Metrics: {metrics_str}")


def setup_training_logger(
    output_dir: Union[str, Path]
) -> tuple[logging.Logger, StructuredLogger]:
    """Setup loggers for training"""
    output_dir = Path(output_dir)

    main_log_file = output_dir / "training.log"
    main_logger = setup_logger(
        name="training", log_file=main_log_file, level=logging.INFO
    )

    metrics_log_file = output_dir / "metrics.jsonl"
    structured_logger = StructuredLogger(
        name="training_metrics", log_file=metrics_log_file
    )

    return main_logger, structured_logger


def configure_root_logger(level: int = logging.WARNING) -> None:
    """Configure root logger to suppress noisy third-party logs"""
    logging.getLogger().setLevel(level)

    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3",
        "transformers.tokenization_utils",
        "transformers.configuration_utils",
        "transformers.modeling_utils",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def cleanup_old_logs(log_dir: Union[str, Path], days: int = 30) -> None:
    """Clean up log files older than specified days"""
    import time

    log_dir = Path(log_dir)
    if not log_dir.exists():
        return

    cutoff_time = time.time() - (days * 24 * 60 * 60)

    for log_file in log_dir.glob("*.log*"):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
        except Exception:
            pass
