import logging
"""
Logger Utility Script

This module provides utility functions for setting up and configuring Python loggers.
It allows users to define logging levels either interactively via input or programmatically
using strings. It also ensures that loggers have a console handler if none exist.

Functions:
- set_logging_level_user(logger): Prompt the user to set the logging level for a logger.
- set_logging_level_str(logger, logLevel): Set the logging level of a logger using a string.
- build_terminal_logger(logLevel, logger_name): Create and configure a logger with the given name and level.
"""

def build_terminal_logger(logger_name, logLevel):
    """
    Create and configure a terminal logger with a given name and log level.

    Parameters:
        logger_name (str): The name of the logger.
        logLevel (str): The logging level as a string ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG').

    Returns:
        logging.Logger: A fully configured logger with a console handler.

    Raises:
        ValueError: If the provided logLevel string is invalid.
    """
    # Map string levels to logging constants
    LOG_LEVELS = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG
    }

    level = LOG_LEVELS.get(logLevel.upper())
    if level is None:
        raise ValueError(f"Invalid log level: {logLevel}. Must be one of {list(LOG_LEVELS.keys())}")

    # Get or create the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Add a console handler if none exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        #formatter = logging.Formatter(
        #    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        #)
        #console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
