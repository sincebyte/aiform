"""FastAPI：表单预测 API + 静态演示页。"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import PredictRequest
from app.predict import _predict_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"

_settings = get_settings()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if _settings.mcp_connections_json and _settings.mcp_connections_json.is_file():
        try:
            from app.history import prewarm_mcp_sql_executor

            await prewarm_mcp_sql_executor(_settings)
            logger.info("MCP SQL 执行器预热完成")
        except Exception:
            logger.warning(
                "MCP SQL 执行器预热失败，首次 predict 时仍会尝试连接",
                exc_info=True,
            )
    yield


app = FastAPI(title="Aiform", version="0.1.0", lifespan=_lifespan)

_origins = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/predict")
async def api_predict(body: PredictRequest):
    async def event_stream():
        try:
            async for event in _predict_stream(_settings, body):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("predict failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if FRONTEND.is_dir():
    app.mount("/sdk", StaticFiles(directory=str(FRONTEND / "sdk")), name="sdk")
    demo_dir = FRONTEND / "demo"

    @app.get("/demo")
    async def demo_index():
        index = demo_dir / "index.html"
        if not index.is_file():
            raise HTTPException(404)
        return FileResponse(index)

    app.mount("/demo-static", StaticFiles(directory=str(demo_dir)), name="demo-static")
