from abc import ABC, abstractmethod
from typing import Tuple, Any
import tensorflow as tf

class BaseDataLoader(ABC):
    """Abstract base class for all data loaders"""
    
    @abstractmethod
    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and return train, validation, and test datasets"""
        pass

    @abstractmethod
    def preprocess_data(self, data: Any) -> Any:
        """Preprocess the data"""
        pass

