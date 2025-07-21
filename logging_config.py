import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(log_folder: str = "logs", log_level: str = "INFO"):
    """
    Set up logging configuration to save logs to a folder with rotation.

    Args:
        log_folder: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"anomaly_detection_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler with rotation (max 10MB per file, keep 5 backup files)
            RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            ),
        ],
    )

    # Create a logger for this application
    logger = logging.getLogger("AnomalyDetection")
    logger.setLevel(getattr(logging, log_level.upper()))

    return logger
