from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os
import time

# Check if deepface is installed, otherwise provide installation instructions
try:
    from deepface import DeepFace
    deepface_available = True
except ImportError:
    deepface_available = False
    print("DeepFace library not found. Install it with: pip install deepface")

app = Flask(__name__)

# Function to create the model if loading fails
def create_ocean_model():
    model = Sequential([
        Dense(16, input_dim=6, activation='relu', name='input_layer'),  # Input: one-hot encoded emotions (6 categories)
        Dense(8, activation='relu'),
        Dense(5, activation='sigmoid')  # Output: OCEAN traits (5 traits)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Try to load the model, recreate if fails
try:
    model = tf.keras.models.load_model('finalocean.h5')
    print("Existing model loaded successfully.")
    model_available = True
except Exception as e:
    print(f"Error loading model: {e}")
    print("Creating a new model...")
    model = create_ocean_model()
    
    # Sample training data
    emotions = ['happy', 'surprise', 'neutral', 'sad', 'anger', 'fear']
    inputs = np.array([
        [1, 0, 0, 0, 0, 0],  # happy
        [0, 1, 0, 0, 0, 0],  # surprise
        [0, 0, 1, 0, 0, 0],  # neutral
        [0, 0, 0, 1, 0, 0],  # sad
        [0, 0, 0, 0, 1, 0],  # anger
        [0, 0, 0, 0, 0, 1],  # fear
    ])
    outputs = np.array([
        [0.7, 0.6, 0.8, 0.7, 0.3],  # OCEAN for 'happy'
        [0.8, 0.5, 0.7, 0.6, 0.4],  # OCEAN for 'surprise'
        [0.5, 0.5, 0.5, 0.5, 0.5],  # OCEAN for 'neutral'
        [0.4, 0.5, 0.3, 0.6, 0.8],  # OCEAN for 'sad'
        [0.5, 0.6, 0.4, 0.2, 0.9],  # OCEAN for 'anger'
        [0.4, 0.6, 0.3, 0.2, 0.9],  # OCEAN for 'fear'
    ])
    
    # Train the model with validation split
    model.fit(inputs, outputs, epochs=50, verbose=1, validation_split=0.2)
    
    # Save the newly trained model
    model.save('finalocean.h5')
    print("New model trained and saved.")
    model_available = True

# Emotion to OCEAN mapping (for fallback)
emotion_to_ocean = {
    'happy': {'O': 0.7, 'C': 0.6, 'E': 0.8, 'A': 0.7, 'N': 0.3},
    'sad': {'O': 0.4, 'C': 0.5, 'E': 0.3, 'A': 0.6, 'N': 0.8},
    'angry': {'O': 0.5, 'C': 0.6, 'E': 0.4, 'A': 0.2, 'N': 0.9},
    'surprise': {'O': 0.8, 'C': 0.5, 'E': 0.7, 'A': 0.6, 'N': 0.4},
    'neutral': {'O': 0.5, 'C': 0.5, 'E': 0.5, 'A': 0.5, 'N': 0.5},
    'fear': {'O': 0.4, 'C': 0.6, 'E': 0.3, 'A': 0.2, 'N': 0.9},
    'disgust': {'O': 0.3, 'C': 0.5, 'E': 0.4, 'A': 0.3, 'N': 0.7}
}

# Store aggregated OCEAN scores
aggregated_scores = []

def map_emotion_to_vector(emotion):
    """Convert emotion to a one-hot vector."""
    emotions = ['happy', 'surprise', 'neutral', 'sad', 'anger', 'fear']
    try:
        # Handle possible naming differences
        if emotion == 'angry':
            emotion = 'anger'
        return [1 if emotion == e else 0 for e in emotions]
    except Exception:
        # Default to neutral if emotion not recognized
        return [0, 0, 1, 0, 0, 0]

def map_emotion_to_ocean(dominant_emotion):
    try:
        # Normalize the emotion name
        emotion = dominant_emotion.lower()
        if emotion == 'angry':
            emotion = 'anger'
        
        # Convert one-hot encoded emotion to OCEAN traits using the model
        if model_available:
            emotion_vector = np.array([map_emotion_to_vector(emotion)])
            ocean_traits = model.predict(emotion_vector)[0]
            
            return {
                'O': float(ocean_traits[0]),
                'C': float(ocean_traits[1]),
                'E': float(ocean_traits[2]),
                'A': float(ocean_traits[3]),
                'N': float(ocean_traits[4])
            }
        else:
            # Fallback to predefined mapping
            return emotion_to_ocean.get(emotion, emotion_to_ocean['neutral'])
    except Exception as e:
        print(f"Model prediction failed: {e}")
        # Fallback to predefined mapping
        return emotion_to_ocean.get(dominant_emotion.lower(), emotion_to_ocean['neutral'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not deepface_available:
        def error_frame():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "DeepFace not installed. Run: pip install deepface"
            cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            return (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return Response(error_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def generate_frames():
        global aggregated_scores
        
        # Try to access the camera
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")
        except Exception as e:
            # Handle camera access error
            print(f"Camera error: {e}")
            def error_message():
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera not available", (100, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return error_message()
        
        # Load Haar cascade for face detection
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            face_cascade = None
            print("Warning: Could not load face cascade classifier")

        frame_count = 0
        last_analysis_time = time.time()
        last_emotion = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame_count += 1
            current_time = time.time()
            
            # Display frame with or without analysis
            try:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces less frequently to improve performance
                if face_cascade is not None:
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                else:
                    # If face_cascade failed to load, assume a face in the center
                    h, w = frame.shape[:2]
                    faces = [(int(w/4), int(h/4), int(w/2), int(h/2))]

                for (x, y, w, h) in faces:
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Extract face region and analyze less frequently (every 15 frames)
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if frame_count % 15 == 0 or current_time - last_analysis_time > 3.0:
                        try:
                            # Use DeepFace to detect emotion
                            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                            
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            
                            dominant_emotion = analysis.get('dominant_emotion', None)
                            last_emotion = dominant_emotion
                            last_analysis_time = current_time
                            
                            if dominant_emotion:
                                # Map emotion to OCEAN traits
                                ocean_scores = map_emotion_to_ocean(dominant_emotion.lower())
                                aggregated_scores.append(ocean_scores)

                                # Display emotion on frame
                                cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y-25), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                # Display personality ratios on frame
                                y_offset = 0
                                for i, (trait, score) in enumerate(ocean_scores.items()):
                                    text = f"{trait}: {score:.2f}"
                                    cv2.putText(frame, text, (x, y-10-y_offset), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
                                    y_offset += 20
                        except Exception as e:
                            print(f"Error analyzing face: {e}")
                    elif last_emotion:
                        # Display the last detected emotion
                        cv2.putText(frame, f"Emotion: {last_emotion}", (x, y-25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
            except Exception as e:
                print(f"Error processing frame: {e}")

            # Encode the frame to send to the client
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze', methods=['POST'])
def analyze():
    global aggregated_scores
    if aggregated_scores:
        # Calculate average OCEAN scores
        final_scores = {}
        for trait in ['O', 'C', 'E', 'A', 'N']:
            values = [score.get(trait, 0.5) for score in aggregated_scores]
            final_scores[trait] = np.mean(values)
        
        # Create pie chart for results
        create_pie_chart(final_scores)
        
        aggregated_scores = []  # Clear scores for next session
        return redirect(url_for('results'))
    else:
        return "No data captured. Please try again.", 400

def create_pie_chart(ocean_scores):
    traits = {
        'O': 'Openness',
        'C': 'Conscientiousness',
        'E': 'Extraversion',
        'A': 'Agreeableness',
        'N': 'Neuroticism'
    }
    
    # Prepare data for pie chart
    labels = [f"{traits[trait]} ({trait}): {score:.2f}" for trait, score in ocean_scores.items()]
    sizes = list(ocean_scores.values())
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('OCEAN Personality Traits', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/ocean_pie_chart.png')
    plt.close()

@app.route('/results')
def results():
    return render_template('results.html', image_url='/static/ocean_pie_chart.png')

if __name__ == '__main__':
    # Ensure static and template directories exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5001)