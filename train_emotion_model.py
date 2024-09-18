from data_preprocessing import load_and_preprocess_fer2013
from model_architecture import build_emotion_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load and preprocess the FER2013 dataset
csv_file_path = 'dataset/fer2013.csv'
faces, emotions = load_and_preprocess_fer2013(csv_file_path)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)

# Build the model using the architecture from model.py
model = build_emotion_model()

# Summary of the model
model.summary()

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,  # Rotate images up to 10 degrees
    width_shift_range=0.1,  # Shift images horizontally by 10% of the width
    height_shift_range=0.1,  # Shift images vertically by 10% of the height
    zoom_range=0.1,  # Random zoom by 10%
    horizontal_flip=True  # Randomly flip images horizontally
)

# Fit the data generator on the training data
datagen.fit(X_train)

# Train the model using augmented data
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=50, validation_data=(X_val, y_val))

# Save the trained model
model.save('model/emotion_model.h5')
