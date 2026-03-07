import logging
import os
from datetime import datetime


# Create logs directory at runtime working directory
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Timestamped log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# Configure root logging once
if not logging.getLogger().handlers:
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)