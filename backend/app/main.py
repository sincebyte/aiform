"""FastAPI：表单预测 API + 静态演示页。"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import PredictRequest, PredictResponse
from app.predict import predict_form

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"

app = FastAPI(title="Aiform", version="0.1.0")
_settings = get_settings()

_origins = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/predict", response_model=PredictResponse)
async def api_predict(body: PredictRequest) -> PredictResponse:
    try:
        return await predict_form(_settings, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


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
