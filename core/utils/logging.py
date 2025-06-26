import os
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(level=logging.DEBUG):
    """
    Set up comprehensive logging with both console and file output.
    Enhanced for debugging A2A communication issues.
    """
    # Check if handlers are already configured
    if logging.getLogger().handlers:
        return
    
    # Create logs directory if it doesn't exist
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)-20s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler (simplified for readability)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.INFO)  # Less verbose for console
    root_logger.addHandler(console_handler)
    
    # File handler (detailed for debugging)
    log_file = log_dir / f"cherryai_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)  # Full detail for file
    root_logger.addHandler(file_handler)
    
    # Special handler for A2A communication debugging
    a2a_log_file = log_dir / f"a2a_debug_{datetime.now().strftime('%Y%m%d')}.log"
    a2a_handler = logging.FileHandler(a2a_log_file, encoding='utf-8')
    a2a_handler.setFormatter(detailed_formatter)
    a2a_handler.setLevel(logging.DEBUG)
    
    # Add A2A handler to specific loggers
    a2a_loggers = [
        'core.plan_execute.a2a_executor',
        'a2a_servers.pandas_server',
        'httpx',
        'a2a'
    ]
    
    for logger_name in a2a_loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(a2a_handler)
        logger.setLevel(logging.DEBUG)
    
    logging.info("🔧 Enhanced logging configured successfully")
    logging.info(f"📁 Log files: {log_file} and {a2a_log_file}")

def log_a2a_request(method: str, url: str, data: dict = None, headers: dict = None):
    """Log A2A request details for debugging"""
    logger = logging.getLogger('a2a.request')
    logger.debug("=" * 60)
    logger.debug(f"🌐 A2A REQUEST: {method} {url}")
    if headers:
        logger.debug(f"📋 Headers: {headers}")
    if data:
        logger.debug(f"📦 Data: {data}")
    logger.debug("=" * 60)

def log_a2a_response(status_code: int, response_data: dict = None, error: str = None):
    """Log A2A response details for debugging"""
    logger = logging.getLogger('a2a.response')
    logger.debug("=" * 60)
    if error:
        logger.error(f"❌ A2A RESPONSE ERROR: {status_code} - {error}")
    else:
        logger.debug(f"✅ A2A RESPONSE: {status_code}")
    if response_data:
        logger.debug(f"📦 Response Data: {response_data}")
    logger.debug("=" * 60)

def log_ai_function(response: str, file_name: str, log: bool = True, log_path: str = './logs/', overwrite: bool = True):
    """
    Logs the response of an AI function to a file.
    
    Parameters
    ----------
    response : str
        The response of the AI function.
    file_name : str
        The name of the file to save the response to.
    log : bool, optional
        Whether to log the response or not. The default is True.
    log_path : str, optional
        The path to save the log file. The default is './logs/'.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists. The default is True.
        - If True, the file will be overwritten. 
        - If False, a unique file name will be created.
    
    Returns
    -------
    tuple
        The path and name of the log file.    
    """
    
    if log:
        # Ensure the directory exists
        os.makedirs(log_path, exist_ok=True)

        # file_name = 'data_wrangler.py'
        file_path = os.path.join(log_path, file_name)

        if not overwrite:
            # If file already exists and we're NOT overwriting, we create a new name
            if os.path.exists(file_path):
                # Use an incremental suffix (e.g., data_wrangler_1.py, data_wrangler_2.py, etc.)
                # or a time-based suffix if you prefer.
                base_name, ext = os.path.splitext(file_name)
                i = 1
                while True:
                    new_file_name = f"{base_name}_{i}{ext}"
                    new_file_path = os.path.join(log_path, new_file_name)
                    if not os.path.exists(new_file_path):
                        file_path = new_file_path
                        file_name = new_file_name
                        break
                    i += 1

        # Write the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response)

        print(f"      File saved to: {file_path}")
        
        return (file_path, file_name)
    
    else:
        return (None, None)