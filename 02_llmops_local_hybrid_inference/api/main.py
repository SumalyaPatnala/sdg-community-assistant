import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving.model_router import ModelRouter
from monitoring.logger import log_event
from monitoring.metrics import MetricsStore


app = FastAPI(
    title="LLMOps Platform for Local and Hybrid Inference",
    version="1.0.0",
    description="Production-style API for summarization, extraction, classification, routing, and monitoring."
)

router = ModelRouter()
metrics = MetricsStore()


class InferenceRequest(BaseModel):
    task: str = Field(..., examples=["summarization", "extraction", "classification"])
    text: str
    model_preference: str = Field("auto", examples=["auto", "local", "remote"])
    max_tokens: int = 512


class InferenceResponse(BaseModel):
    request_id: str
    task: str
    model_used: str
    latency_ms: float
    output: str


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "local_model_available": router.local_available(),
        "remote_model_available": router.remote_available()
    }


@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest):
    request_id = str(uuid.uuid4())
    start = time.time()

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        output, model_used = router.generate(
            task=req.task,
            text=req.text,
            model_preference=req.model_preference,
            max_tokens=req.max_tokens
        )

        latency_ms = round((time.time() - start) * 1000, 2)

        record = {
            "request_id": request_id,
            "task": req.task,
            "model_used": model_used,
            "latency_ms": latency_ms,
            "input_chars": len(req.text),
            "status": "success"
        }

        log_event(record)
        metrics.add(record)

        return InferenceResponse(
            request_id=request_id,
            task=req.task,
            model_used=model_used,
            latency_ms=latency_ms,
            output=output
        )

    except Exception as e:
        latency_ms = round((time.time() - start) * 1000, 2)
        record = {
            "request_id": request_id,
            "task": req.task,
            "model_used": "none",
            "latency_ms": latency_ms,
            "input_chars": len(req.text),
            "status": "error",
            "error": str(e)
        }
        log_event(record)
        metrics.add(record)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics():
    return metrics.summary()


@app.get("/models")
def get_models():
    return router.model_registry()
