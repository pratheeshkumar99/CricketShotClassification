from abc import ABC, abstractmethod
from typing import Dict, Any
import tensorflow as tf

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def build(self) -> None:
        """Build the model architecture"""
        pass

    @abstractmethod
    def compile(self, **kwargs) -> None:
        """Compile the model"""
        pass

    @abstractmethod
    def train(self, train_data: tf.data.Dataset, val_data: tf.data.Dataset, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    def evaluate(self, test_data: tf.data.Dataset) -> Dict[str, float]:
        """Evaluate the model"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model"""
        pass