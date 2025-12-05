"""
@author: Zhu
@date: 2025/03/21
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.responses import ORJSONResponse
from feature import feature_service
from starlette.middleware.cors import CORSMiddleware

from loguru import logger

from feature.feature_service import FeatureService
from feature.logger import init_log, log_details

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    # 移除默认的 logging 处理器
    import logging
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    # 定义日志格式
    init_log()
    
    logger.info("Starting server...")

    app = FastAPI(
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    #注册中间件
    app.middleware("http")(log_details)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def get_feature_service() -> FeatureService:
        return FeatureService()
    
    @app.post("/img_file/")
    async def img_file(file: UploadFile = File(...), service: FeatureService = Depends(get_feature_service)):
        return {"image": file.filename, "feature": await service.feature(await file.read())}

    return app


def main():
    import uvicorn
    uvicorn.run("feature.main:create_app", host="0.0.0.0", port=8000, reload=True, factory=True)


if __name__ == '__main__':
    main()
