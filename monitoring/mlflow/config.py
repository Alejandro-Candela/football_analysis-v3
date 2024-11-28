from pathlib import Path

# Base paths
BASE_DIR = Path.cwd()
LAB_DIR = BASE_DIR / "lab"
DATA_DIR = BASE_DIR / "data"

# Training paths
TRAINING_DIR = LAB_DIR / "training"
DATA_TRAIN_PATH = DATA_DIR / "train_data"

# Model defaults
DEFAULT_MODEL_NAME = "yolo11n.pt"
MODELS_DIR = LAB_DIR / "models"

def get_experiment_dir(experiment_id: str) -> Path:
    """Get the directory for a specific experiment."""
    return TRAINING_DIR / f"experiment_{experiment_id:03d}"

def create_experiment_structure(experiment_id: str) -> dict[str, Path]:
    """Create and return the experiment directory structure."""
    exp_dir = get_experiment_dir(experiment_id)
    
    paths = {
        'root': exp_dir,
        'config': exp_dir / 'config.yaml',
        'best_weights': exp_dir / 'best.pt',
        'last_weights': exp_dir / 'last.pt',
        'logs': exp_dir / 'logs',
        'artifacts': exp_dir / 'artifacts'
    }
    
    # Create directories
    for path in paths.values():
        if isinstance(path, Path) and not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            
    return paths

