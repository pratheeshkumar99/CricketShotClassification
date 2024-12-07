# scripts/train_all.py
import subprocess
from pathlib import Path

def train_all_models():
    config_dir = Path("configs")
    models = ["b0", "b1", "b2"]
    
    for model in models:
        config_file = config_dir / f"efficientnet_{model}_config.json"
        print(f"\nTraining EfficientNet-{model.upper()}")
        print("=" * 50)
        subprocess.run(["python", "train.py", "--config", str(config_file)])

if __name__ == "__main__":
    train_all_models()