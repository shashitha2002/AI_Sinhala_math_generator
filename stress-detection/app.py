"""
Flask Web Application for Real-time Facial Emotion Recognition
Uses webcam to detect emotions in real-time and displays results in a web browser
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
import threading
import os
import threading
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Face Mesh (Tasks API)
model_path = os.path.abspath('face_landmarker.task')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5)
detector = vision.FaceLandmarker.create_from_options(options)

app = Flask(__name__)

# Load the pre-trained emotion detection model
model = load_model('model.h5')
face_cascade = cv2.CascadeClassifier('HaarcascadeclassifierCascadeClassifier.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Global variables for emotion tracking
emotion_data = {
    'emotion': 'Neutral',
    'confidence': 0,
    'faces_detected': 0
}

# Global variables for webcam capture
camera = None
lock = threading.Lock()


class VideoCamera:
    """Camera object to capture video frames from webcam"""
    def __init__(self):
        try:
            self.video = cv2.VideoCapture(0)
            if not self.video.isOpened():
                raise Exception("Could not open webcam")
            
            # Set camera properties for better performance
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_FPS, 30)
            self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test first frame
            ret, frame = self.video.read()
            if not ret:
                raise Exception("Could not read from webcam")
            
            print("✓ Camera initialized successfully - Ready to stream")
            
        except Exception as e:
            print(f"✗ Camera initialization failed: {e}")
            self.video = None
            raise

    def __del__(self):
        if self.video and self.video.isOpened():
            self.video.release()
            print("✓ Camera released")

    def get_frame(self):
        """Capture frame from webcam and detect emotions"""
        if self.video is None or not self.video.isOpened():
            return None
            
        try:
            ret, frame = self.video.read()
            
            if not ret:
                print("⚠ Failed to capture frame from camera")
                return None
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Create a clean copy for emotion detection BEFORE drawing the mesh
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- MediaPipe Face Mesh Processing (Tasks API) ---
            try:
                # Convert to RGB (MediaPipe requirement)
                # Create MP Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Detect
                detection_result = detector.detect(mp_image)
                
                # Draw landmarks manually
                if detection_result.face_landmarks:
                    # Enforce single face for mesh drawing
                    for face_landmarks in detection_result.face_landmarks[:1]:
                        for landmark in face_landmarks:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            # Draw small white dots for all mesh points
                            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

                        # Feature indices
                        LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
                        RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 382, 381, 380, 374, 373, 390, 249]
                        LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                        RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
                        NOSE = [1, 2, 98, 327, 195, 197]
                        MOUTH = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]

                        # Helper function to get bounding box from indices
                        def draw_feature_box(indices, color):
                            try:
                                landmark_points = []
                                for idx in indices:
                                    lm = face_landmarks[idx]
                                    px = int(lm.x * frame.shape[1])
                                    py = int(lm.y * frame.shape[0])
                                    landmark_points.append((px, py))
                                
                                if landmark_points:
                                    x_coords = [p[0] for p in landmark_points]
                                    y_coords = [p[1] for p in landmark_points]
                                    min_x, max_x = min(x_coords), max(x_coords)
                                    min_y, max_y = min(y_coords), max(y_coords)
                                    
                                    # Add padding
                                    padding = 5
                                    cv2.rectangle(frame, (min_x - padding, min_y - padding), 
                                                (max_x + padding, max_y + padding), color, 2)
                            except IndexError:
                                pass

                        # Draw bounding boxes for features
                        # Colors in BGR: Blue, Green, Red, Yellow
                        draw_feature_box(LEFT_EYE, (255, 0, 0))       # Blue for Eyes
                        draw_feature_box(RIGHT_EYE, (255, 0, 0))
                        draw_feature_box(LEFT_EYEBROW, (0, 255, 0))   # Green for Eyebrows
                        draw_feature_box(RIGHT_EYEBROW, (0, 255, 0))
                        draw_feature_box(NOSE, (0, 0, 255))           # Red for Nose
                        draw_feature_box(MOUTH, (0, 255, 255))        # Yellow for Mouth
            except Exception as e:
                print(f"Face Mesh Error: {e}")
            # --------------------------------------
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Keep only the largest face (closest to camera)
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1]
                print(f"Detected {len(faces)} face(s) - Processing largest only")

            # Update faces detected count (after filtering)
            emotion_data['faces_detected'] = len(faces)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
                # Extract ROI (Region of Interest)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                
                # Preprocess the ROI
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    
                    # Predict emotion
                    try:
                        prediction = model.predict(roi, verbose=0)[0]
                        emotion_idx = prediction.argmax()
                        emotion_label = emotion_labels[emotion_idx]
                        confidence = prediction[emotion_idx] * 100
                        
                        # Update global emotion data
                        emotion_data['emotion'] = str(emotion_label)
                        emotion_data['confidence'] = float(confidence)
                        emotion_data['faces_detected'] = int(len(faces))
                        
                        # Put text on frame
                        label_text = f'{emotion_label} ({confidence:.1f}%)'
                        cv2.putText(frame, label_text, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        print(f"Emotion: {label_text}")
                    except Exception as e:
                        print(f"Error predicting emotion: {e}")
                        cv2.putText(frame, 'Error', (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, 'No Face', (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Add header text
            cv2.putText(frame, 'Real-time Emotion Detection', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add frame info
            if len(faces) == 0:
                cv2.putText(frame, 'No faces detected - Position face towards camera', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Encode frame to JPEG with quality control
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                print("Failed to encode frame")
                return None
            
            return jpeg.tobytes()
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None


def gen(camera):
    """Generator function to stream video frames in MJPEG format"""
    try:
        while True:
            with lock:
                frame = camera.get_frame()
                if frame is None:
                    continue
                
                # Proper MJPEG boundary and headers
                yield b'--frame\r\n'
                yield b'Content-Type: image/jpeg\r\n'
                yield b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                yield frame
                yield b'\r\n'
    except GeneratorExit:
        print("Generator exit")
        pass
    except Exception as e:
        print(f"Error in generator: {e}")


@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    global camera
    
    if camera is None:
        camera = VideoCamera()
    
    response = Response(gen(camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame',
                        headers={
                            'Cache-Control': 'no-cache, no-store, must-revalidate',
                            'Pragma': 'no-cache',
                            'Expires': '0'
                        })
    return response


@app.route('/stop')
def stop_camera():
    """Stop camera route"""
    global camera
    if camera is not None:
        del camera
        camera = None
    return 'Camera stopped'


@app.route('/get_emotion_data')
def get_emotion_data():
    """Get current emotion data"""
    return jsonify(emotion_data)


if __name__ == '__main__':
    # Run the Flask app
    # Set debug=False for production
    # Set host='0.0.0.0' to allow external connections
    app.run(debug=True, host='localhost', port=5000, threaded=True)
