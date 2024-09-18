import cv2
import numpy as np
from tensorflow.keras.models import load_model


class FindValues:
    @staticmethod
    def get_frame_width_height(frame):
        """
        Returns the width and height of the frame.
        Args:
            frame: The input image or video frame.
        Returns:
            (width, height): Tuple of frame width and height.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")
        return frame.shape[1], frame.shape[0]


class FaceDetection:
    def __init__(self, face_cascade, eye_cascade, emotion_model, emotion_labels):
        """
        Initializes the FaceDetection class with cascades and emotion model.
        Args:
            face_cascade: Haar cascade for face detection.
            eye_cascade: Haar cascade for eye detection.
            emotion_model: Pre-trained Keras model for emotion recognition.
            emotion_labels: List of emotion labels corresponding to model outputs.
        """
        if not all([face_cascade, eye_cascade, emotion_model]):
            raise ValueError("One or more detection models are missing.")

        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        self.emotion_model = emotion_model
        self.emotion_labels = emotion_labels

    def detect_and_predict(self, frame, scale_factor=1.2, min_neighbors=5, line_thickness=2):
        """
        Detects faces, eyes and predicts emotions on a given frame.
        Args:
            frame: The input frame from the video feed.
            scale_factor: Scaling factor for Haar cascade face detection.
            min_neighbors: Minimum neighbors for face detection.
            line_thickness: Thickness of the rectangle lines drawn around faces.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a white rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), line_thickness)
            face_roi_gray = gray[y:y + h, x:x + w]
            # Detect eyes within the face region
            self._detect_eyes(face_roi_gray, frame, x, y, w, h, line_thickness)
            # Predict emotion for the detected face
            self._predict_emotion(face_roi_gray, frame, x, y)

    def _detect_eyes(self, face_roi_gray, frame, x, y, w, h, line_thickness):
        """
        Detects eyes within the face region.
        Args:
            face_roi_gray: Grayscale region of the face.
            frame: Original frame where rectangles will be drawn.
            x, y, w, h: Coordinates and size of the detected face.
            line_thickness: Thickness of rectangle around eyes.
        """
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            # Draw a white rectangle around the eyes
            cv2.rectangle(frame[y:y + h, x:x + w], (ex, ey), (ex + ew, ey + eh), (255, 255, 255), line_thickness)

    def _predict_emotion(self, face_roi_gray, frame, x, y):
        """
        Predicts the emotion based on the detected face region.
        Args:
            face_roi_gray: Grayscale region of the face.
            frame: Original frame where the predicted emotion label will be drawn.
            x, y: Coordinates of the detected face where text will be drawn.
        """
        # Resize face region to 48x48 for emotion prediction
        face_roi_resized = cv2.resize(face_roi_gray, (48, 48)).astype('float32') / 255.0
        face_roi_resized = np.expand_dims(np.expand_dims(face_roi_resized, axis=0), axis=-1)

        # Predict emotion using the loaded model
        emotion_prediction = self.emotion_model.predict(face_roi_resized)
        max_index = np.argmax(emotion_prediction)
        predicted_emotion = self.emotion_labels[max_index]

        # Show the predicted emotion label on the frame bottom
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


class LiveFaceDetection:
    def __init__(self, model_path='model/emotion_model.h5', frame_skip=1):
        """
        Initializes the live face detection system.
        Args:
            model_path: Path to the pre-trained emotion recognition model.
            frame_skip: Number of frames to skip before processing for performance optimization.
        """
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.live_face_detection()

    def live_face_detection(self):
        """
        Captures live video feed from the webcam and performs face, eye detection, and emotion prediction.
        """
        cap = cv2.VideoCapture(0)  # Use camera index 0 for most systems
        if not cap.isOpened():
            print("Error: Could not access the camera.")
            return

        print('Camera On')

        # Load Haar Cascades and emotion model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        try:
            emotion_model = load_model(self.model_path)
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            return

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        detector = FaceDetection(face_cascade, eye_cascade, emotion_model, emotion_labels)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            if frame_count % self.frame_skip == 0:
                detector.detect_and_predict(frame)

            cv2.imshow('Emotion Detector', frame)

            if cv2.waitKey(1) == ord('q'):  # Press q to exit
                print('Camera Off')
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    LiveFaceDetection()
