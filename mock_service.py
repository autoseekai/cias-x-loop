"""
Mock SCI Service - Simulates the SCI Training Service

A FastAPI-based mock service that simulates:
- Training task submission
- Task status polling
- Result retrieval

Run with: uvicorn mock_service:app --reload --port 8000
"""

import asyncio
import random
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np


# ==================== Data Models ====================

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainRequest(BaseModel):
    """Training request model"""
    experiment_id: str
    forward_model: Dict[str, Any]
    reconstruction: Dict[str, Any]
    training: Dict[str, Any]
    uncertainty_quantification: Optional[Dict[str, Any]] = None


class TaskResponse(BaseModel):
    """Task creation response"""
    task_id: str
    experiment_id: str
    status: TaskStatus
    created_at: str


class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    experiment_id: str
    status: TaskStatus
    progress: float
    message: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class MetricsResponse(BaseModel):
    """Experiment metrics"""
    psnr: float
    ssim: float
    coverage: float
    latency: float
    memory: float
    training_time: float
    convergence_epoch: int


class ResultResponse(BaseModel):
    """Task result response"""
    task_id: str
    experiment_id: str
    status: TaskStatus
    metrics: Optional[MetricsResponse] = None
    artifacts: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    completed_at: Optional[str] = None


# ==================== Task Storage ====================

class TaskStore:
    """In-memory task storage"""

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, experiment_id: str, config: Dict[str, Any]) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "task_id": task_id,
            "experiment_id": experiment_id,
            "config": config,
            "status": TaskStatus.PENDING,
            "progress": 0.0,
            "message": "Task queued",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "metrics": None,
            "artifacts": None,
            "error_message": None,
        }
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.tasks.get(task_id)

    def update_task(self, task_id: str, updates: Dict[str, Any]):
        if task_id in self.tasks:
            self.tasks[task_id].update(updates)


# Global task store
task_store = TaskStore()


# ==================== Mock Training Logic ====================

def generate_mock_metrics(config: Dict[str, Any]) -> MetricsResponse:
    """
    Generate mock metrics based on configuration.
    Simulates realistic behavior where certain configs perform better.
    """
    # Base metrics
    base_psnr = 25.0
    base_ssim = 0.80

    # Extract config parameters
    recon = config.get("reconstruction", {})
    forward = config.get("forward_model", {})

    num_stages = recon.get("num_stages", 5)
    num_features = recon.get("num_features", 64)
    compression_ratio = forward.get("compression_ratio", 16)

    # Simulate performance effects
    # More stages generally improve quality
    stage_bonus = (num_stages - 5) * 0.3

    # More features improve quality but diminishing returns
    feature_bonus = np.log2(num_features / 32) * 0.5

    # Higher compression is harder to reconstruct
    cr_penalty = (compression_ratio - 8) * 0.1

    # Calculate metrics with randomness
    psnr = base_psnr + stage_bonus + feature_bonus - cr_penalty + np.random.randn() * 1.5
    ssim = base_ssim + 0.02 * (stage_bonus + feature_bonus - cr_penalty) / 2 + np.random.randn() * 0.03

    # Clamp values
    psnr = max(20.0, min(35.0, psnr))
    ssim = max(0.65, min(0.98, ssim))

    # Latency increases with model complexity
    latency = 30 + num_stages * 5 + num_features * 0.1 + np.random.randn() * 5

    # Training time correlates with complexity
    training_time = 0.5 + num_stages * 0.2 + num_features * 0.01 + np.random.rand() * 0.5

    return MetricsResponse(
        psnr=round(psnr, 2),
        ssim=round(ssim, 4),
        coverage=round(0.85 + np.random.rand() * 0.1, 2),
        latency=round(max(10, latency), 1),
        memory=int(1024 + num_features * 16 + np.random.rand() * 512),
        training_time=round(training_time, 2),
        convergence_epoch=int(20 + np.random.randint(-5, 15))
    )


async def simulate_training(task_id: str, config: Dict[str, Any]):
    """
    Simulate the training process with progress updates.
    """
    # Mark as running
    task_store.update_task(task_id, {
        "status": TaskStatus.RUNNING,
        "started_at": datetime.now().isoformat(),
        "message": "Training started"
    })

    # Simulate training phases
    phases = [
        (0.1, "Initializing model"),
        (0.3, "Loading dataset"),
        (0.5, "Training epoch 1-25"),
        (0.7, "Training epoch 26-40"),
        (0.9, "Training epoch 41-50"),
        (1.0, "Evaluating model")
    ]

    # Random total training time (1-5 seconds for mock)
    total_time = random.uniform(1.0, 5.0)

    for progress, message in phases:
        await asyncio.sleep(total_time / len(phases))
        task_store.update_task(task_id, {
            "progress": progress,
            "message": message
        })

    # Simulate occasional failures (5% chance)
    if random.random() < 0.05:
        task_store.update_task(task_id, {
            "status": TaskStatus.FAILED,
            "completed_at": datetime.now().isoformat(),
            "error_message": "Training diverged: gradient explosion detected",
            "message": "Training failed"
        })
        return

    # Generate mock metrics
    metrics = generate_mock_metrics(config)

    # Generate mock artifacts
    artifacts = {
        "checkpoint_path": f"/checkpoints/{task_id}/model_best.pth",
        "training_log_path": f"/logs/{task_id}/training.log",
        "sample_reconstructions": [
            f"/samples/{task_id}/recon_{i}.png" for i in range(3)
        ],
        "figure_scripts": [
            f"/figures/{task_id}/plot_metrics.py"
        ],
        "metrics_history": {
            "psnr": [metrics.psnr - random.uniform(2, 5) + i * random.uniform(0.1, 0.2)
                     for i in range(50)],
            "ssim": [metrics.ssim - random.uniform(0.05, 0.1) + i * random.uniform(0.001, 0.003)
                     for i in range(50)]
        }
    }

    task_store.update_task(task_id, {
        "status": TaskStatus.COMPLETED,
        "completed_at": datetime.now().isoformat(),
        "metrics": metrics.model_dump(),
        "artifacts": artifacts,
        "message": "Training completed successfully"
    })


# ==================== FastAPI Application ====================

app = FastAPI(
    title="Mock SCI Training Service",
    description="A mock service simulating SCI reconstruction training",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Mock SCI Training Service",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok"}


@app.post("/train", response_model=TaskResponse)
async def submit_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Submit a new training task.

    The task will be processed asynchronously.
    Use the returned task_id to poll for status and retrieve results.
    """
    config = {
        "forward_model": request.forward_model,
        "reconstruction": request.reconstruction,
        "training": request.training,
        "uncertainty_quantification": request.uncertainty_quantification
    }

    task_id = task_store.create_task(request.experiment_id, config)

    # Start training in background
    background_tasks.add_task(simulate_training, task_id, config)

    return TaskResponse(
        task_id=task_id,
        experiment_id=request.experiment_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now().isoformat()
    )


@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the current status of a training task.
    """
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=task["task_id"],
        experiment_id=task["experiment_id"],
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at")
    )


@app.get("/tasks/{task_id}/result", response_model=ResultResponse)
async def get_task_result(task_id: str):
    """
    Get the result of a completed training task.
    """
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed yet. Current status: {task['status']}"
        )

    metrics = None
    if task.get("metrics"):
        metrics = MetricsResponse(**task["metrics"])

    return ResultResponse(
        task_id=task["task_id"],
        experiment_id=task["experiment_id"],
        status=task["status"],
        metrics=metrics,
        artifacts=task.get("artifacts"),
        error_message=task.get("error_message"),
        completed_at=task.get("completed_at")
    )


@app.get("/tasks")
async def list_tasks(limit: int = 10, offset: int = 0):
    """
    List all tasks with pagination.
    """
    all_tasks = list(task_store.tasks.values())
    return {
        "total": len(all_tasks),
        "tasks": all_tasks[offset:offset + limit]
    }


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a pending or running task.
    """
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed task")

    task_store.update_task(task_id, {
        "status": TaskStatus.FAILED,
        "completed_at": datetime.now().isoformat(),
        "error_message": "Task cancelled by user",
        "message": "Cancelled"
    })

    return {"status": "cancelled", "task_id": task_id}


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
