"""
error_utilities.py
Author: Roger Erismann

Purpose:
---------
Centralized error handling utilities for logging and debugging during pipeline operations.

Key Features:
-------------
- `handle_error()`: Generic function to log and return user-friendly error messages.
- `handle_errors()`: Decorator to wrap functions with standardized try/except behavior and helpful logging.

Usage Examples:
---------------
# Use the decorator to wrap pipeline steps:
@handle_errors("loading CSV", "Verify the file path and format are correct")
def load_data(...): ...

# Use the function manually inside exception blocks:
try:
    ...
except Exception as e:
    msg = handle_error(e, "model training", "Check model config and data", logger)
"""

import functools

def handle_error(error: Exception, message: str, tip: str, logger) -> str:
    """
    Logs an error using the provided logger and returns a user-friendly message.

    Args:
        error: The caught exception.
        message: A short description of the operation being attempted.
        tip: Suggestion for the user to resolve the issue.
        logger: Logger instance (from calling module).

    Returns:
        A user-safe error message string.
    """
    full_message = f"Error during {message}: {str(error)}"
    logger.error(full_message, exc_info=True)
    return f"Something went wrong while {message}. {tip}"


def handle_errors(message: str, tip: str):
    """
    Decorator for handling exceptions in any function.

    Args:
        message: Description of the function's purpose.
        tip: User-facing suggestion for recovery.

    Usage:
        @handle_errors("loading shapefile", "Check if the file path and format are correct.")
        def load_data(...): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = kwargs.get("logger")
            try:
                return func(*args, **kwargs)
            except RuntimeError as re:
                if logger:
                    logger.warning(f"Handled error (already wrapped): {re}")
                raise  # Don't re-wrap!

            # Handle new errors
            except Exception as e:
                if logger:
                    logger.error(f"Unhandled error during {message}: {e}", exc_info=True)
                raise RuntimeError(f"Something went wrong while {message}. {tip}")
        return wrapper
    return decorator
