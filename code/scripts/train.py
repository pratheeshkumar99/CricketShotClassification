import argparse
import json
from pathlib import Path
from typing import Dict, Any
from src.data.data_loader import VideoDataLoader
from src.models.efficient_net_variants import create_efficient_net
from src.utils.visualization import Visualizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return json.dump(f)
def main():
    args = parse_args()
    config = load_config(args.config)
    
    print("Initializing data loader...")
    data_loader = VideoDataLoader(config)
    train_ds, val_ds, test_ds = data_loader.load_data()
    
    print(f"Creating EfficientNet-{config['backbone']} model...")
    model = create_efficient_net(config)
    model.build()
    model.compile()
    
    print("Starting training...")
    history = model.train(train_ds, val_ds)
    
    print("Evaluating model...")
    results = model.evaluate(test_ds)
    
    # Save results
    visualizer = Visualizer(config['output_dir'])
    visualizer.plot_training_history(history)
    visualizer.save_results(results)
    
    print(f"Training completed. Results saved to {config['output_dir']}")

if __name__ == '__main__':
    main()