from pathlib import Path
import yaml
import logging
import shutil
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from monitoring.mlflow.config import (
    create_experiment_structure,
    DATA_TRAIN_PATH,
    DEFAULT_MODEL_NAME,
    MODELS_DIR,
    TRAINING_DIR
)
from training.train_utils import (
    get_next_experiment_id,
    save_experiment_config,
    verify_dataset_structure,
    train_model,
    evaluate_model,
    load_model,
    clear_dataset_cache
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline."""
    try:
        # Get next experiment ID and create directory structure
        experiment_id = get_next_experiment_id(TRAINING_DIR)
        experiment_paths = create_experiment_structure(experiment_id)
        
        # Load model and verify dataset
        model = load_model(model_directory=MODELS_DIR, model_name=DEFAULT_MODEL_NAME)
        data_yaml = DATA_TRAIN_PATH / 'data.yaml'
        
        # Clear cache before verification and training
        clear_dataset_cache(data_yaml)
        
        if not verify_dataset_structure(data_yaml=data_yaml):
            raise ValueError("Dataset structure validation failed")
        
        # Train model
        metrics, params = train_model(
            model=model,
            data_yaml=data_yaml,
            experiment_paths=experiment_paths,
            epochs=2
        )
        
        # Save experiment configuration
        save_experiment_config(experiment_paths['config'], params, DEFAULT_MODEL_NAME)
        
        # Save training artifacts
        shutil.copy(
            str(experiment_paths['logs'] / 'train' / 'results.png'),
            str(experiment_paths['artifacts'] / 'training_results.png')
        )
        
        # Evaluate model
        eval_metrics = evaluate_model(
            model_path=experiment_paths['best_weights'],
            data_yaml=data_yaml
        )
        metrics.update(eval_metrics)
        
        # Save metrics to logs
        metrics_path = experiment_paths['logs'] / 'metrics.yaml'
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
            
        logger.info(f"Training completed successfully. Experiment ID: {experiment_id}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()