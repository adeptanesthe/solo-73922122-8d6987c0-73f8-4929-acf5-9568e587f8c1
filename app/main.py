"""FastAPI 入口：上传参考/待检 WAV，返回冲次核对结果。"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio import AudioValidationError, MAX_DURATION_SEC, SAMPLE_RATE
from .sensitivity import SensitivityInputError
from .service import MAX_UPLOAD_BYTES, Store

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/workspace/data"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="冲压机冲次声音核对", version="1.1.0")
store = Store(DATA_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/data/uploads/{name}")
def get_upload(name: str) -> FileResponse:
    # 动态读取 store 的上传目录，更换 DATA_DIR 或测试夹具后仍可回传录音
    if "/" in name or ".." in name or not name.endswith(".wav"):
        raise HTTPException(status_code=400, detail="非法文件名")
    path = store.uploads_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="录音不存在")
    return FileResponse(path, media_type="audio/wav")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/result/last")
def last_result() -> dict:
    result = store.get_last()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无有效检查结果")
    return result


@app.delete("/api/result/last")
def clear_result() -> dict[str, str]:
    store.clear()
    return {"status": "cleared"}


_WAV_MAX_BYTES = MAX_UPLOAD_BYTES


async def _read_wav_upload(file: UploadFile, label: str) -> bytes:
    if not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail=f"{label}必须是 .wav 文件")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{label}文件为空")
    if len(raw) > _WAV_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label}文件过大；仅支持 {SAMPLE_RATE}Hz/16bit/单声道、"
                f"不超过 {MAX_DURATION_SEC:.0f} 秒的 WAV"
            ),
        )
    return raw


@app.post("/api/inspect")
async def inspect(
    reference: UploadFile = File(..., description="正常循环参考 WAV"),
    test: UploadFile = File(..., description="待检循环 WAV"),
) -> dict:
    ref_raw = await _read_wav_upload(reference, "参考录音")
    test_raw = await _read_wav_upload(test, "待检录音")

    try:
        # 任何 AudioValidationError 都会在 store 写入前抛出，
        # 因此无效音频不会覆盖上一次有效结果
        return store.inspect(ref_raw, test_raw)
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sensitivity/last")
def last_sensitivity() -> dict:
    result = store.get_last_sensitivity()
    if result is None:
        raise HTTPException(status_code=404, detail="暂无灵敏度试验记录")
    return result


@app.delete("/api/sensitivity/last")
def clear_sensitivity() -> dict[str, str]:
    store.clear_sensitivity()
    return {"status": "cleared"}


@app.post("/api/sensitivity")
async def sensitivity(
    idle: UploadFile | None = File(None, description="空转录音 WAV（保持原音量）"),
    sample: UploadFile | None = File(None, description="单冲样音 WAV（≤80ms）"),
    impact_time: str | None = Form(
        None, description="样音中冲击的时刻（秒）"
    ),
    trial_times: str | None = Form(
        None, description="3～6 个试验时刻（秒），JSON 数组或逗号分隔"
    ),
) -> dict:
    # 所有输入错误都交给 service 统一处理：撤下旧推荐并返回 400
    if idle is None:
        store.retract_sensitivity("缺少空转录音文件")
        raise HTTPException(status_code=400, detail="缺少空转录音文件")
    if sample is None:
        store.retract_sensitivity("缺少样音文件")
        raise HTTPException(status_code=400, detail="缺少样音文件")

    try:
        idle_raw = await _read_wav_upload(idle, "空转录音")
        sample_raw = await _read_wav_upload(sample, "样音")
        return store.run_sensitivity(
            idle_raw, sample_raw, impact_time, trial_times
        )
    except HTTPException as exc:
        # read_wav_upload 的格式/大小错误同样视为输入有误
        store.retract_sensitivity(str(exc.detail))
        raise
    except SensitivityInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
