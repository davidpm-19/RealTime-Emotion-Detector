import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


# Function to load and preprocess the FER2013 dataset
def load_and_preprocess_fer2013(csv_file_path, width=48, height=48, num_classes=7):
    """
    Load and preprocess the FER2013 dataset from a CSV file.

    Args:
    - csv_file_path (str): Path to the FER2013 CSV file.
    - width (int): Width of the image (default is 48).
    - height (int): Height of the image (default is 48).
    - num_classes (int): Number of emotion classes (default is 7).

    Returns:
    - faces (numpy array): Preprocessed images as numpy arrays.
    - emotions (numpy array): One-hot encoded emotion labels.
    """

    # Load the FER CSV file
    data = pd.read_csv(csv_file_path)

    # Extract image pixels and emotions
    pixels = data['pixels'].tolist()
    emotions = data['emotion'].values

    # Preprocess the pixel data
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]  # Convert pixel values to integers
        face = np.asarray(face).reshape(width, height)  # Reshape into a 48x48 image
        face = face.astype('float32')
        faces.append(face)

    # Convert the face list into a numpy array
    faces = np.array(faces)
    faces = np.expand_dims(faces, -1)  # Add channel dimension (for grayscale)

    # Normalize pixel values to be between 0 and 1
    faces /= 255.0

    # One-hot encode the emotions
    emotions = to_categorical(emotions, num_classes)

    return faces, emotions
