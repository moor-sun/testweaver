# utils/logging.py
import logging
import os

LOG_LEVEL = os.getenv("TESTWEAVER_LOG_LEVEL", "DEBUG").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger("testweaver")
