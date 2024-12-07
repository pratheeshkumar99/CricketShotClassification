import argparse
from pathlib import Path
import json
from src.data.data_loader import VideoDataLoader
from src.models.video_classifier import VideoClassifier
from src.utils.visualization import Visualizer
from typing import Dict, Any 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return json.load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize data loader
    data_loader = VideoDataLoader(config)
    train_ds, val_ds, test_ds = data_loader.load_data()
    
    # Initialize and train model
    model = VideoClassifier(config)
    model.build()
    model.compile()
    
    history = model.train(train_ds, val_ds)
    
    # Evaluate and save results
    results = model.evaluate(test_ds)
    
    # Save visualizations
    visualizer = Visualizer(config['output_dir'])
    visualizer.plot_training_history(history)
    visualizer.save_results(results)
    
    print(f"Training completed. Results saved to {config['output_dir']}")

if __name__ == '__main__':
    main()
