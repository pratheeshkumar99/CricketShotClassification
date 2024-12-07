from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from .base_efficient_net import BaseEfficientNet
from typing import Dict, Any

class EfficientNetB0Model(BaseEfficientNet):
    """EfficientNetB0 implementation"""
    def get_backbone(self):
        return EfficientNetB0

class EfficientNetB1Model(BaseEfficientNet):
    """EfficientNetB1 implementation"""
    def get_backbone(self):
        return EfficientNetB1

class EfficientNetB2Model(BaseEfficientNet):
    """EfficientNetB2 implementation"""
    def get_backbone(self):
        return EfficientNetB2

def create_efficient_net(config: Dict[str, Any]) -> BaseEfficientNet:
    """Factory function to create the appropriate EfficientNet model"""
    model_map = {
        'B0': EfficientNetB0Model,
        'B1': EfficientNetB1Model,
        'B2': EfficientNetB2Model
    }
    model_class = model_map.get(config['backbone'])
    if model_class is None:
        raise ValueError(f"Unsupported backbone: {config['backbone']}")
    return model_class(config)