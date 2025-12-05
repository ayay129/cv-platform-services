from fastapi import Request, Response
from loguru import logger
import time

class LoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next) -> Response:
        start_time = time.time()
        
        # 记录请求信息
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host}:{request.client.port}"
        )
        
        try:
            response = await call_next(request)
        except Exception as e:
            # 记录异常信息
            logger.exception(
                f"Request failed: {str(e)}"
            )
            raise e from None
        finally:
            # 记录响应时间
            process_time = (time.time() - start_time) * 1000
            logger.info(
                f"Response in {process_time:.2f}ms"
            )

        return response