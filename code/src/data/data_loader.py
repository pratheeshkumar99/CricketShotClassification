from typing import Dict, Tuple
import tensorflow as tf
from pathlib import Path
from ..base.base_data import BaseDataLoader
from .video_processor import VideoProcessor
import random
import numpy as np

class VideoDataLoader(BaseDataLoader):
    """Handles loading and preprocessing of video data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processor = VideoProcessor(output_size=tuple(config['input_shape'][:2]))
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validate that all required data paths exist"""
        required_paths = ['train_path', 'val_path', 'test_path']
        for path in required_paths:
            if not Path(self.config[path]).exists():
                raise ValueError(f"Path does not exist: {self.config[path]}")

    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        train_ds = self._create_dataset(self.config['train_path'], is_training=True)
        val_ds = self._create_dataset(self.config['val_path'])
        test_ds = self._create_dataset(self.config['test_path'])
        return train_ds, val_ds, test_ds

    def _create_dataset(self, path: str, is_training: bool = False) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            self._generator_function(path, is_training),
            output_signature=self._get_signature()
        )
        return self.preprocess_data(dataset)

    def preprocess_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return (dataset
                .batch(self.config['batch_size'])
                .prefetch(tf.data.AUTOTUNE)
                .cache())

    def _generator_function(self, path: str, is_training: bool):
        def generator():
            path_obj = Path(path)
            video_paths = list(path_obj.glob('*/*.mp4'))
            if is_training:
                random.shuffle(video_paths)
            for video_path in video_paths:
                frames = self.processor.process_video(
                    str(video_path), 
                    self.config['num_frames']
                )
                yield np.stack([f.data for f in frames]), self._get_label(video_path)
        return generator

    def _get_signature(self) -> Tuple[tf.TensorSpec, tf.TensorSpec]:
        input_height, input_width = self.config['input_shape'][:2]
        return (
            tf.TensorSpec(shape=(self.config['num_frames'], input_height, input_width, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.uint8)
        )

    def _get_label(self, video_path: Path) -> int:
        return self.config['class_mapping'][video_path.parent.name]