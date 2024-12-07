from dataclasses import dataclass
from typing import Tuple, List
import tensorflow as tf
import cv2
import numpy as np
from typing import List

@dataclass
class VideoFrame:
    """Data class to store video frame information"""
    data: np.ndarray
    timestamp: float
    frame_number: int

class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self, output_size: Tuple[int, int] = (224, 224)):
        self.output_size = output_size
        self._frame_processor = FrameProcessor(output_size)

    def process_video(self, video_path: str, n_frames: int) -> List[VideoFrame]:
        frames = []
        with VideoCapture(video_path) as cap:
            for frame_number in range(n_frames):
                frame = cap.read_frame()
                if frame is not None:
                    processed_frame = self._frame_processor.process(frame)
                    frames.append(VideoFrame(
                        data=processed_frame,
                        timestamp=cap.get_timestamp(),
                        frame_number=frame_number
                    ))
        return frames

class FrameProcessor:
    """Handles individual frame processing"""
    
    def __init__(self, output_size: Tuple[int, int]):
        self.output_size = output_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        frame = tf.image.convert_image_dtype(frame, tf.uint8)
        frame = tf.image.resize_with_pad(frame, *self.output_size)
        return frame.numpy()

class VideoCapture:
    """Context manager for video capture operations"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def read_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_timestamp(self) -> float:
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)
