import logging
import sys
from datetime import datetime
from typing import Any, Dict

# Create a custom logger
logger = logging.getLogger("rag_chatbot")
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)

# Also configure uvicorn logs
logging.getLogger("uvicorn").handlers = [console_handler]
logging.getLogger("uvicorn.access").handlers = [console_handler]
logging.getLogger("uvicorn.error").handlers = [console_handler]

def log_api_call(endpoint: str, method: str, params: Dict[str, Any] = None, response_status: int = 200):
    """
    Log API calls for monitoring and debugging
    """
    logger.info(f"API CALL: {method} {endpoint} - Status: {response_status}")
    if params:
        logger.debug(f"API CALL PARAMS: {params}")

def log_error(error: Exception, context: str = ""):
    """
    Log errors with context
    """
    logger.error(f"ERROR in {context}: {str(error)}", exc_info=True)

def log_info(message: str):
    """
    Log informational messages
    """
    logger.info(message)

def log_debug(message: str):
    """
    Log debug messages
    """
    logger.debug(message)

def log_warning(message: str):
    """
    Log warning messages
    """
    logger.warning(message)