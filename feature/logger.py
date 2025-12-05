"""
@author: Zhu
@date: 2025/03/27
"""

import sys
from loguru import logger
from fastapi import Request
import time


def init_log():
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    

async def log_details(req: Request, call_next):

    start_time = time.time()

    try:
        res = await call_next(req)
    except Exception as e:
        logger.error(f"Request Error | {str(e)}")
        raise
    process_time = (time.time() - start_time) * 1000
    process_time = round(process_time, 2)
    logger.info(f"Method: {req.method}, Url: {req.url}, Host: {req.client.host}, Process Time: {process_time}ms")

    return res