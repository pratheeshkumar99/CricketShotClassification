import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import EfficientNetB0
import tempfile
import shutil

# Load pre-trained EfficientNetB0 without the top layer to use as a feature extractor
st.set_page_config(layout="wide")

# Define class labels
classes = {'cover': 0, 'defense': 1, 'flick': 2, 'hook': 3, 'late_cut': 4, 'lofted': 5, 'pull': 6, 'square_cut': 7, 'straight': 8, 'sweep': 9}

# Function to load the model
def load_model(weights_path):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Set the base model as non-trainable
    base_model.trainable = False

# Define the full model using a Sequential model
    model = models.Sequential([
    # Apply EfficientNetB0 to each frame of the video
        layers.TimeDistributed(base_model, input_shape=(None, 224, 224, 3)),
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),

    # Use GRU layers to capture temporal relationships
        layers.GRU(256, return_sequences=True),
        layers.GRU(128),

    # Dense layers for classification
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.load_weights(weights_path)
    return model

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded.
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.uint8)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame.numpy()

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=1):
    """
    Extracts frames sequentially from the start of the video file, with a specified step between frames.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image (height, width).
      frame_step: Number of frames to skip between extracted frames.

    Returns:
      A NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    result = []
    src = cv2.VideoCapture(str(video_path))

    src.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start from the first frame

    # Attempt to read the first frame
    ret, frame = src.read()
    if ret:
        frame = format_frames(frame, output_size)
        result.append(frame)
    else:
        # If the first frame can't be read, append a zero frame and exit
        result.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8))

    # Read subsequent frames with the specified frame_step
    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            # Append a zero-like frame if no more frames can be read
            result.append(np.zeros_like(result[0]))

    src.release()

    # Convert the list of frames to a NumPy array and adjust color channels from BGR to RGB
    result = np.array(result)[..., [2, 1, 0]]

    return result

# Function to classify video
def classify_video(video_path, model, frame_count, class_labels):
    # Process the video file to get the frames
    frames = frames_from_video_file(video_path, frame_count)

    # Add batch dimension if the model expects it
    frames = np.expand_dims(frames, axis=0)

    # Use the model to predict the class probabilities
    predictions = model.predict(frames)
    print("Raw predictions:", predictions)

    # Convert predictions to class labels
    predicted_class_idx = np.argmax(predictions, axis=1)[0]  # Get the index of the max class score
    print("Predicted class index:", predicted_class_idx)
    
    # Get the class name using the predicted index
    predicted_class_name = list(class_labels.keys())[list(class_labels.values()).index(predicted_class_idx)]
    
    # Calculate the confidence percentage of the predicted class
    confidence = predictions[0][predicted_class_idx] * 100  # Assuming softmax output, multiply by 100 for percentage
    print("Confidence (%): {:.2f}%".format(confidence))

    return predicted_class_name, confidence

# Streamlit user interface
st.title('Cricket Shot Classification and Similarity Checker')

# Load model 
model = load_model('model_weights.h5')

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmpfile:
            shutil.copyfileobj(uploaded_file, tmpfile)
            tmp_path = tmpfile.name
        return tmp_path
    except Exception as e:
        print(f"Error saving uploaded file to temp directory: {e}")
        return None

col1, col2 = st.columns(2)
class1 = conf1 = class2 = conf2 = None

with col1:
    video1 = st.file_uploader("Upload first video", type=["mp4", "avi"], key="video1")
    if video1:
        st.video(video1)
        video1_path = save_uploaded_file(video1)
        class1, conf1 = classify_video(video1_path, model, 30, classes)
        st.success(f"First video classified as {class1} with confidence {conf1:.2f}%")

with col2:
    video2 = st.file_uploader("Upload second video", type=["mp4", "avi", "move"], key="video2")
    if video2:
        st.video(video2)
        video2_path = save_uploaded_file(video2)
        class2, conf2 = classify_video(video2_path, model, 30, classes)
        st.success(f"Second video classified as {class2} with confidence {conf2:.2f}%")

if st.button('Compare Videos'):
    if video1 is not None and video2 is not None and class1 == class2:
        # Extract features for similarity check
        feature_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
        features1 = feature_model.predict(np.expand_dims(frames_from_video_file(video1_path, 30), axis=0))
        features2 = feature_model.predict(np.expand_dims(frames_from_video_file(video2_path, 30), axis=0))
        
        # Compute cosine similarity
        dot_product = np.dot(features1, features2.T)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        similarity = dot_product / (norm1 * norm2)
        
        st.success(f"Similarity between videos: {similarity[0][0] * 100:.2f}%")
    elif class1 is not None and class2 is not None and class1 != class2:
        st.write("Videos are of different classes; similarity is not computed.")
    else:
        st.write("Please upload both videos to compare.")
