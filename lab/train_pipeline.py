from pathlib import Path
import yaml
from typing import Dict, Any, Tuple
import shutil
import logging
from ultralytics import YOLO
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_next_experiment_id() -> int:
    """Get the next available experiment ID."""
    existing_experiments = [
        int(p.name.split('_')[1])
        for p in TRAINING_DIR.glob('experiment_*')
        if p.name.split('_')[1].isdigit()
    ]
    return max(existing_experiments, default=0) + 1

def save_experiment_config(config_path: Path, params: Dict[str, Any]) -> None:
    """Save experiment configuration to YAML file."""
    config = {
        'parameters': params,
        'timestamp': datetime.now().isoformat(),
        'model_name': DEFAULT_MODEL_NAME
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def verify_dataset_structure(data_yaml: Path) -> bool:
    """Verify dataset structure matches yaml configuration."""
    try:
        if not data_yaml.exists():
            logger.error(f"data.yaml not found at: {data_yaml}")
            return False
            
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Verify required keys
        required_keys = ['path', 'train', 'val', 'test', 'names']
        if missing_keys := [key for key in required_keys if key not in data_config]:
            logger.error(f"Missing required keys in data.yaml: {missing_keys}")
            return False
            
        # Verify paths exist
        base_path = data_yaml.parent
        paths = {
            'train': base_path / 'train' / 'images',
            'val': base_path / 'validation' / 'images',
            'test': base_path / 'test' / 'images'
        }
        
        for name, path in paths.items():
            if not path.exists() or not list(path.glob('*.jpg')) + list(path.glob('*.png')):
                logger.error(f"Missing or empty {name} directory: {path}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error verifying dataset: {str(e)}")
        return False

def train_model(
    model: YOLO,
    data_yaml: Path,
    experiment_paths: Dict[str, Path],
    epochs: int = 100
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Train the YOLO model and return metrics."""
    try:
        # Start training
        logger.info(f"Starting training with data config: {data_yaml}")
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=640,
            batch=16,
            device='cuda',
            workers=8,
            project=str(experiment_paths['logs']),
            name="train",
            exist_ok=True,
            pretrained=True,
            verbose=True
        )
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
            'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
            'precision': float(results.results_dict.get('metrics/precision(B)', 0.0)),
            'recall': float(results.results_dict.get('metrics/recall(B)', 0.0)),
            'fitness': float(results.results_dict.get('fitness', 0.0))
        }
        
        # Save training parameters
        params = {
            'epochs': epochs,
            'batch_size': 16,
            'image_size': 640,
            'device': 'cuda',
            'model_type': 'YOLOv8n',
            'data_yaml': str(data_yaml)
        }
        
        # Save weights
        shutil.copy(
            str(experiment_paths['logs'] / 'train' / 'weights' / 'best.pt'),
            str(experiment_paths['best_weights'])
        )
        shutil.copy(
            str(experiment_paths['logs'] / 'train' / 'weights' / 'last.pt'),
            str(experiment_paths['last_weights'])
        )
        
        return metrics, params
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def evaluate_model(model_path: Path, data_yaml: Path) -> Dict[str, float]:
    """Evaluate the model on validation and test data."""
    try:
        model = YOLO(str(model_path))
        
        # Validate on validation set
        logger.info("Evaluating on validation set...")
        val_results = model.val(data=str(data_yaml), split='val')
        
        # Validate on test set
        logger.info("Evaluating on test set...")
        test_results = model.val(data=str(data_yaml), split='test')
        
        return {
            'val_mAP50': float(val_results.results_dict.get('metrics/mAP50(B)', 0)),
            'val_mAP50-95': float(val_results.results_dict.get('metrics/mAP50-95(B)', 0)),
            'test_mAP50': float(test_results.results_dict.get('metrics/mAP50(B)', 0)),
            'test_mAP50-95': float(test_results.results_dict.get('metrics/mAP50-95(B)', 0))
        }
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def load_model(model_directory: Path, model_name: str) -> YOLO:
    """Load or download YOLO model."""
    try:
        model_directory.mkdir(parents=True, exist_ok=True)
        model_path = model_directory / model_name
        
        if not model_path.exists():
            logger.info(f"Downloading model {model_name}...")
            model = YOLO(model_name)
            model.save(str(model_path))
        else:
            logger.info(f"Loading existing model from {model_path}")
            model = YOLO(str(model_path))
            
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def clear_dataset_cache(data_yaml: Path) -> None:
    """Clear all .cache files in the dataset directories."""
    try:
        base_path = data_yaml.parent
        cache_patterns = [
            'train/labels.cache',
            'validation/labels.cache',
            'test/labels.cache',
            '*.cache'  # Catch any cache files in the root directory
        ]
        
        # Also check the v2 directory path
        v2_path = Path(str(base_path).replace('v3', 'v2'))
        
        for directory in [base_path, v2_path]:
            logger.info(f"Clearing cache files in: {directory}")
            for pattern in cache_patterns:
                cache_files = list(directory.glob(pattern))
                for cache_file in cache_files:
                    try:
                        cache_file.unlink()
                        logger.info(f"Deleted cache file: {cache_file}")
                    except Exception as e:
                        logger.warning(f"Could not delete {cache_file}: {str(e)}")
                        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise

def main():
    """Main training pipeline."""
    try:
        # Get next experiment ID and create directory structure
        experiment_id = get_next_experiment_id()
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
        save_experiment_config(experiment_paths['config'], params)
        
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