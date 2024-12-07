from typing import Dict, Any, Tuple
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from ..base.base_model import BaseModel

class VideoClassifier(BaseModel):
    """Video classification model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.history = None
        self._backbone = self._get_backbone()

    def _get_backbone(self):
        backbones = {
            'B0': EfficientNetB0,
            'B1': EfficientNetB1,
            'B2': EfficientNetB2
        }
        if self.config['backbone'] not in backbones:
            raise ValueError(f"Unsupported backbone: {self.config['backbone']}")
        return backbones[self.config['backbone']]

    def build(self) -> None:
        base_model = self._backbone(
            include_top=False,
            weights='imagenet',
            input_shape=self.config['input_shape']
        )
        base_model.trainable = False

        self.model = models.Sequential([
            layers.TimeDistributed(
                base_model, 
                input_shape=(None,) + self.config['input_shape']
            ),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            *self._build_temporal_layers(),
            *self._build_classification_layers()
        ])

    def _build_temporal_layers(self) -> Tuple[layers.Layer, ...]:
        return (
            layers.GRU(self.config['gru_units'][0], return_sequences=True),
            layers.GRU(self.config['gru_units'][1])
        )

    def _build_classification_layers(self) -> Tuple[layers.Layer, ...]:
        return (
            layers.Dense(1024, activation='relu'),
            layers.Dropout(self.config['dropout_rate']),
            layers.Dense(self.config['num_classes'], activation='softmax')
        )

    def compile(self, **kwargs) -> None:
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['learning_rate']
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_data: tf.data.Dataset, 
              val_data: tf.data.Dataset, **kwargs) -> Dict[str, Any]:
        callbacks = self._get_callbacks()
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            **kwargs
        )
        return self.history.history

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                self.config['model_path'],
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience']
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=self.config['patience'] // 2
            )
        ]

    def evaluate(self, test_data: tf.data.Dataset) -> Dict[str, float]:
        test_loss, test_accuracy = self.model.evaluate(test_data)
        return {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy)
        }

    def save(self, path: str) -> None:
        self.model.save_weights(path)

    def load(self, path: str) -> None:
        self.model.load_weights(path)