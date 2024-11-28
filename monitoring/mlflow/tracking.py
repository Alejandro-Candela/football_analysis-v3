from pathlib import Path
import logging
from typing import Dict, Any

from monitoring.mlflow.setup import setup_mlflow, log_model_metrics, register_model
from monitoring.mlflow.config import BEST_MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def track_experiment(
    metrics: Dict[str, float],
    params: Dict[str, Any],
    model_path: Path,
    artifacts: Dict[str, str] | None = None
) -> None:
    """
    Track an experiment using MLflow.
    
    Args:
        metrics: Dictionary of metrics (e.g., mAP, precision, recall)
        params: Dictionary of parameters used in training
        model_path: Path to the saved model
        artifacts: Dictionary of additional artifacts to log
    """
    try:
        # Initialize MLflow
        setup_mlflow()

        # Log the experiment
        run_id = log_model_metrics(
            metrics=metrics,
            params=params,
            model_path=model_path,
            artifacts=artifacts
        )

        # Register the model if it's the best model
        if model_path == BEST_MODEL_PATH:
            register_model(run_id, model_path)

        logger.info("Successfully tracked experiment with run_id: %s", run_id)

    except Exception as e:
        logger.error(f"Error tracking experiment: {str(e)}")
        raise
