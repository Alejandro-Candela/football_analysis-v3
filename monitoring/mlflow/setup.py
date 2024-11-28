import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from monitoring.mlflow.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    BEST_MODEL_PATH
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow() -> None:
    """Initialize MLflow configuration."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create experiment if it doesn't exist
    try:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    except Exception:
        logger.info(f"Experiment {MLFLOW_EXPERIMENT_NAME} already exists")
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def log_model_metrics(
    metrics: Dict[str, float],
    params: Dict[str, Any],
    model_path: Optional[Path] = None,
    artifacts: Optional[Dict[str, str]] = None
) -> str:
    """
    Log model metrics, parameters, and artifacts to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        params: Dictionary of parameter names and values
        model_path: Path to the model file
        artifacts: Dictionary of artifact names and paths
    
    Returns:
        run_id: The ID of the MLflow run
    """
    with mlflow.start_run() as run:
        # Log metrics and parameters
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        
        # Log model file if provided
        if model_path and model_path.exists():
            mlflow.log_artifact(str(model_path))
        
        # Log additional artifacts if provided
        if artifacts:
            for name, path in artifacts.items():
                if Path(path).exists():
                    mlflow.log_artifact(path, name)
        
        return run.info.run_id

def register_model(run_id: str, model_path: Path) -> None:
    """
    Register a model in the MLflow Model Registry.
    
    Args:
        run_id: The ID of the MLflow run
        model_path: Path to the model file
    """
    client = MlflowClient()
    
    try:
        # Register the model
        model_uri = f"runs:/{run_id}/{model_path.name}"
        mv = client.create_model_version(
            name=MLFLOW_MODEL_NAME,
            source=model_uri,
            run_id=run_id
        )
        
        # Transition the model to Production if it's the best model
        if model_path == BEST_MODEL_PATH:
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME,
                version=mv.version,
                stage="Production"
            )
            
        logger.info(f"Registered model version {mv.version} for {MLFLOW_MODEL_NAME}")
        
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")