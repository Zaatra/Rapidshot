import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

# Default log formats
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log levels
DEFAULT_CONSOLE_LEVEL = logging.WARNING
DEFAULT_FILE_LEVEL = logging.DEBUG

# Default log file location
DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), ".rapidshot", "logs")
DEFAULT_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "rapidshot.log")

# Maximum log file size (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Number of backup log files to keep
BACKUP_COUNT = 5

# Environment variable to control log level
LOG_LEVEL_ENV_VAR = "RAPIDSHOT_LOG_LEVEL"
LOG_FILE_ENV_VAR = "RAPIDSHOT_LOG_FILE"

# Global logger registry to avoid duplicate configuration
_loggers = {}

def setup_logging(
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for RapidShot.
    
    Args:
        console_level: Log level for console handler
        file_level: Log level for file handler
        log_file: Path to log file
        log_format: Log format string
        date_format: Date format string
        
    Returns:
        Logger: Root logger for RapidShot
    """
    # Create logger
    logger = logging.getLogger("rapidshot")
    
    # If logger is already configured, return it
    if logger.handlers:
        return logger
        
    # Set logger level to the minimum of console and file levels
    logger.setLevel(logging.DEBUG)
    
    # Get log level from environment variable if set
    env_log_level = os.environ.get(LOG_LEVEL_ENV_VAR)
    if env_log_level:
        try:
            env_log_level = env_log_level.upper()
            if env_log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                console_level = getattr(logging, env_log_level)
            elif env_log_level.isdigit():
                console_level = int(env_log_level)
        except (ValueError, AttributeError):
            pass
    
    # Set default values if not provided
    console_level = console_level if console_level is not None else DEFAULT_CONSOLE_LEVEL
    file_level = file_level if file_level is not None else DEFAULT_FILE_LEVEL
    log_format = log_format if log_format is not None else DEFAULT_LOG_FORMAT
    date_format = date_format if date_format is not None else DEFAULT_DATE_FORMAT
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Check if log file is provided or use environment variable
    if log_file is None:
        log_file = os.environ.get(LOG_FILE_ENV_VAR, DEFAULT_LOG_FILE)
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create log directory {log_dir}: {e}")
            return logger
    
    try:
        # Create file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, IOError) as e:
        logger.warning(f"Could not create log file {log_file}: {e}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Name of the module
        
    Returns:
        Logger for the module
    """
    if name in _loggers:
        return _loggers[name]
        
    if name.startswith("rapidshot."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(f"rapidshot.{name}")
        
    _loggers[name] = logger
    return logger


# Initialize logging when module is imported
setup_logging()