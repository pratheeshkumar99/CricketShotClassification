import matplotlib.pyplot as plt
from pathlib import Path
import json

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_history(self, history: dict):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history['accuracy'], label='Training')
        ax1.plot(history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(history['loss'], label='Training')
        ax2.plot(history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()

    def save_results(self, results: dict):
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)