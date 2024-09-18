from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input


def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Build and compile the CNN model for emotion detection.

    Args:
    - input_shape (tuple): Shape of the input images (default is (48, 48, 1)).
    - num_classes (int): Number of emotion classes (default is 7).

    Returns:
    - model (keras.Model): Compiled CNN model.
    """
    model = Sequential()

    # First convolutional layer
    model.add(Input(shape=input_shape))  # Specify input shape
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fourth convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Regularization
    model.add(Dropout(0.25))

    # Flatten the output of the convolutional layers
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))  # Prevent Overfitting

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
