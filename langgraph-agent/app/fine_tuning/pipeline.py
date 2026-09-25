"""
app/fine_tuning/pipeline.py  —  OpenAI Fine-Tuning Pipeline

─────────────────────────────────────────────────────────────────────────────
THIS FILE CONTAINS REAL OPENAI API CALLS.
When USE_SIMULATION=True, the actual API calls are skipped but the code
structure, logic, and explanations are identical to what runs in production.

PIPELINE STEPS:
  1. prepare_dataset()     — validate and split data
  2. upload_dataset()      — upload training file to OpenAI
  3. start_finetuning()    — kick off the fine-tuning job
  4. poll_status()         — wait for job completion (can take hours)
  5. deploy()              — update settings to use fine-tuned model
  6. monitor()             — track performance via Langfuse

HOW FINE-TUNING ACTUALLY WORKS INTERNALLY:
  1. OpenAI receives your JSONL training file
  2. Splits your data into mini-batches
  3. Runs forward pass: model predicts assistant tokens given system+user
  4. Computes cross-entropy loss: how different is prediction from target?
  5. Backpropagation: adjusts model weights to reduce loss
  6. Repeats for N epochs (default 3)
  7. Validates on your val set after each epoch
  8. Returns a new model ID: "ft:gpt-3.5-turbo:your-org:model-name:id"

OVERFITTING DETECTION:
  If training loss keeps decreasing but validation loss increases,
  the model is memorizing training examples instead of generalizing.
  Fix: Reduce epochs, add more diverse data, reduce data repetition.

─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Set to True to simulate (no real API calls, no cost)
# Set to False to run real fine-tuning (costs money, takes hours)
USE_SIMULATION = True


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Upload training file
# ─────────────────────────────────────────────────────────────────────────────

def upload_dataset(file_path: str, simulate: bool = USE_SIMULATION) -> dict:
    """
    Upload a JSONL training file to OpenAI.

    WHAT HAPPENS:
      OpenAI validates the file format and returns a file_id.
      You then use this file_id to start the fine-tuning job.

    REAL COST: Free (just file storage)
    TIME: Seconds to minutes depending on file size

    EXAMPLE RETURN:
      {"file_id": "file-abc123", "filename": "train.jsonl", "bytes": 12345}
    """
    if simulate:
        logger.info("[SIMULATION] upload_dataset: %s", file_path)
        return {
            "file_id": "file-SIMULATED_ID_abc123def456",
            "filename": Path(file_path).name,
            "bytes": Path(file_path).stat().st_size if Path(file_path).exists() else 0,
            "status": "processed",
            "simulated": True,
        }

    # ── REAL CODE (runs when simulate=False) ──────────────────────────────────
    from openai import OpenAI
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
    )

    logger.info("Uploading training file: %s", file_path)
    with open(file_path, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune",   # MUST be "fine-tune" for fine-tuning
        )

    logger.info("File uploaded: id=%s bytes=%d", response.id, response.bytes)
    return {
        "file_id": response.id,
        "filename": response.filename,
        "bytes": response.bytes,
        "status": response.status,
        "simulated": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Start fine-tuning job
# ─────────────────────────────────────────────────────────────────────────────

def start_finetuning(
    training_file_id: str,
    validation_file_id: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    n_epochs: int = 3,
    suffix: str = "hr-assistant",
    simulate: bool = USE_SIMULATION,
) -> dict:
    """
    Start a fine-tuning job on OpenAI.

    PARAMETERS:
      training_file_id   — file ID from upload_dataset()
      validation_file_id — optional val file (enables per-epoch val metrics)
      model              — base model to fine-tune (gpt-3.5-turbo recommended)
      n_epochs           — how many times to train on the full dataset
                           3 = default (good for 100+ examples)
                           1 = fast, less overfitting, less improvement
                           10 = risk of overfitting on small datasets
      suffix             — name suffix for the resulting model

    COST EXAMPLE:
      100 examples × 200 tokens × 3 epochs = 60,000 tokens
      gpt-3.5-turbo: 60,000 / 1M × $8 = $0.48 total

    RETURNS JOB ID immediately. You must poll for completion (hours).

    EXAMPLE RETURN:
      {"job_id": "ftjob-abc123", "status": "validating_files", "model": "..."}
    """
    if simulate:
        logger.info("[SIMULATION] start_finetuning: model=%s epochs=%d", model, n_epochs)
        return {
            "job_id": "ftjob-SIMULATED_abc123",
            "status": "running",
            "model": model,
            "training_file": training_file_id,
            "validation_file": validation_file_id,
            "n_epochs": n_epochs,
            "suffix": suffix,
            "estimated_time_hours": 0.5,
            "simulated": True,
        }

    # ── REAL CODE ─────────────────────────────────────────────────────────────
    from openai import OpenAI
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
    )

    hyperparameters = {"n_epochs": n_epochs}

    kwargs = {
        "training_file": training_file_id,
        "model": model,
        "hyperparameters": hyperparameters,
        "suffix": suffix,
    }
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id

    logger.info("Starting fine-tuning job: %s", kwargs)
    job = client.fine_tuning.jobs.create(**kwargs)

    logger.info(
        "Fine-tuning started: job_id=%s status=%s",
        job.id, job.status
    )
    return {
        "job_id": job.id,
        "status": job.status,
        "model": job.model,
        "training_file": job.training_file,
        "n_epochs": n_epochs,
        "suffix": suffix,
        "created_at": job.created_at,
        "simulated": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Poll for completion
# ─────────────────────────────────────────────────────────────────────────────

def poll_status(job_id: str, simulate: bool = USE_SIMULATION) -> dict:
    """
    Check the status of a fine-tuning job.

    JOB STATUSES:
      validating_files → queued → running → succeeded / failed / cancelled

    TRAINING METRICS (when validation file provided):
      train_loss        — should decrease over epochs
      valid_loss        — should also decrease (if increasing, overfitting!)
      valid_mean_token_accuracy — should increase

    USAGE:
      while True:
          status = poll_status(job_id)
          if status["status"] in ("succeeded", "failed"):
              break
          time.sleep(60)   # poll every minute

    RETURNS:
      {"status": "succeeded", "fine_tuned_model": "ft:gpt-3.5-turbo:org:name:id"}
    """
    if simulate:
        logger.info("[SIMULATION] poll_status: job_id=%s → succeeded", job_id)
        return {
            "job_id": job_id,
            "status": "succeeded",
            "fine_tuned_model": "ft:gpt-3.5-turbo:your-org:hr-assistant:SIMULATED_ID",
            "training_metrics": {
                "train_loss": [1.42, 0.89, 0.63],         # decreasing = good
                "valid_loss": [1.51, 0.95, 0.71],         # both decreasing = not overfitting
                "valid_mean_token_accuracy": [0.62, 0.78, 0.84],  # increasing = learning
            },
            "completed_at": int(time.time()),
            "trained_tokens": 12_000,
            "epochs_completed": 3,
            "simulated": True,
        }

    # ── REAL CODE ─────────────────────────────────────────────────────────────
    from openai import OpenAI
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
    )

    job = client.fine_tuning.jobs.retrieve(job_id)

    # Collect training metrics from events
    events = client.fine_tuning.jobs.list_events(job_id, limit=20)
    metrics = {"events": [{"message": e.message, "created_at": e.created_at} for e in events.data]}

    return {
        "job_id": job.id,
        "status": job.status,
        "fine_tuned_model": job.fine_tuned_model,
        "training_metrics": metrics,
        "trained_tokens": job.trained_tokens,
        "completed_at": job.finished_at,
        "simulated": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Deploy (update config to use fine-tuned model)
# ─────────────────────────────────────────────────────────────────────────────

def deploy(
    fine_tuned_model_id: str,
    env_file: str = ".env",
    simulate: bool = USE_SIMULATION,
) -> dict:
    """
    "Deploy" a fine-tuned model by updating the .env file.

    HOW DEPLOYMENT WORKS:
      Fine-tuned OpenAI models are accessed via the same Chat Completions API.
      You just change the model name from "gpt-3.5-turbo" to your fine-tuned ID.
      No servers to deploy, no containers to rebuild.

      "ft:gpt-3.5-turbo:your-org:hr-assistant:abc123"
       ↑ standard API endpoint, different model ID

    VERSIONING:
      Each fine-tuning job produces a unique model ID. Keep old IDs:
        v1: ft:gpt-3.5-turbo:org:hr-assistant-v1:id1  (baseline)
        v2: ft:gpt-3.5-turbo:org:hr-assistant-v2:id2  (improved with more data)
      Rollback = just change model ID back to v1.
    """
    if simulate:
        logger.info("[SIMULATION] deploy: model=%s", fine_tuned_model_id)
        return {
            "ok": True,
            "deployed_model": fine_tuned_model_id,
            "env_file": env_file,
            "action": "Would update OPENAI_CHAT_MODEL in .env",
            "simulated": True,
            "rollback_command": f"Set OPENAI_CHAT_MODEL=gpt-4o in {env_file} to revert",
        }

    # ── REAL CODE ─────────────────────────────────────────────────────────────
    env_path = Path(env_file)
    if not env_path.exists():
        return {"ok": False, "error": f"{env_file} not found"}

    content = env_path.read_text()
    # Find and replace the OPENAI_CHAT_MODEL line
    import re
    updated = re.sub(
        r"^OPENAI_CHAT_MODEL=.*$",
        f"OPENAI_CHAT_MODEL={fine_tuned_model_id}",
        content,
        flags=re.MULTILINE,
    )
    if "OPENAI_CHAT_MODEL" not in updated:
        updated += f"\nOPENAI_CHAT_MODEL={fine_tuned_model_id}\n"

    env_path.write_text(updated)
    logger.info("Deployed model: %s (restart FastAPI to apply)", fine_tuned_model_id)

    return {
        "ok": True,
        "deployed_model": fine_tuned_model_id,
        "env_file": env_file,
        "action": "Updated OPENAI_CHAT_MODEL in .env — restart FastAPI to apply",
        "simulated": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Complete pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    output_dir: str = "data/fine_tuning",
    model: str = "gpt-3.5-turbo",
    n_epochs: int = 3,
    simulate: bool = USE_SIMULATION,
) -> dict:
    """
    Run the complete fine-tuning pipeline end-to-end.

    PIPELINE:
      1. prepare_dataset() — build and validate JSONL files
      2. upload_dataset()  — upload to OpenAI
      3. start_finetuning() — kick off job
      4. poll_status()    — wait for completion (simulated: instant)
      5. deploy()          — update .env to use new model

    PRODUCTION USAGE (simulate=False):
      This can take 1-12 hours. Run as a background job, not in a web request.
      Set up a Celery task or a standalone script that runs periodically.

    USAGE:
      result = run_pipeline(simulate=True)   # learn the process, no cost
      result = run_pipeline(simulate=False)  # real fine-tuning (costs money!)
    """
    from app.fine_tuning.dataset_builder import prepare_dataset

    logger.info("=== FINE-TUNING PIPELINE START (simulate=%s) ===", simulate)
    pipeline_start = time.time()

    # Step 1: Prepare dataset
    logger.info("Step 1/5: Preparing dataset...")
    dataset_result = prepare_dataset(output_dir)
    if not dataset_result["ok"]:
        return {"ok": False, "stage": "dataset_preparation", "errors": dataset_result.get("errors")}

    # Step 2: Upload training file
    logger.info("Step 2/5: Uploading training file...")
    upload_result = upload_dataset(dataset_result["train_path"], simulate=simulate)

    # Also upload validation file if available
    val_file_id = None
    if dataset_result.get("val_path"):
        val_upload = upload_dataset(dataset_result["val_path"], simulate=simulate)
        val_file_id = val_upload.get("file_id")

    # Step 3: Start fine-tuning
    logger.info("Step 3/5: Starting fine-tuning job...")
    job_result = start_finetuning(
        training_file_id=upload_result["file_id"],
        validation_file_id=val_file_id,
        model=model,
        n_epochs=n_epochs,
        simulate=simulate,
    )

    # Step 4: Wait for completion
    logger.info("Step 4/5: Polling job status...")
    if simulate:
        status_result = poll_status(job_result["job_id"], simulate=True)
    else:
        # Poll every 60 seconds until done
        logger.info("Fine-tuning started. This can take 1-12 hours. Polling every 60s...")
        while True:
            status_result = poll_status(job_result["job_id"], simulate=False)
            logger.info("Status: %s", status_result["status"])
            if status_result["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(60)

    if status_result["status"] != "succeeded":
        return {
            "ok": False,
            "stage": "fine_tuning",
            "status": status_result["status"],
            "job_id": job_result["job_id"],
        }

    fine_tuned_model = status_result.get("fine_tuned_model", "")

    # Step 5: Deploy
    logger.info("Step 5/5: Deploying model %s...", fine_tuned_model)
    env_path = str(Path(__file__).parents[3] / ".env")
    deploy_result = deploy(fine_tuned_model, env_file=env_path, simulate=simulate)

    total_time = round(time.time() - pipeline_start, 1)
    logger.info("=== FINE-TUNING PIPELINE COMPLETE in %.1fs ===", total_time)

    return {
        "ok": True,
        "simulated": simulate,
        "fine_tuned_model": fine_tuned_model,
        "pipeline_time_seconds": total_time,
        "dataset_stats": dataset_result.get("validation", {}).get("stats", {}),
        "cost_estimate": dataset_result.get("cost_estimate", {}),
        "training_metrics": status_result.get("training_metrics", {}),
        "deployment": deploy_result,
        "next_step": (
            "Model deployed! Restart FastAPI to use it. "
            "Then run evaluation.py to compare vs base model."
        ),
    }


if __name__ == "__main__":
    # Quick test: python -m app.fine_tuning.pipeline
    result = run_pipeline(simulate=True)
    print(json.dumps(result, indent=2, default=str))
