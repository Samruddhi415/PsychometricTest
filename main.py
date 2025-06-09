import os
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from deepface import DeepFace
import re
import google.generativeai as genai
import json

app = Flask(__name__)

# Create necessary directories
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# =============== MBTI MODEL CONFIGURATION ===============
# Configure Gemini API
GEMINI_API_KEY = "api_key"  # Replace with your API key
genai.configure(api_key=GEMINI_API_KEY)

# MBTI Questionnaire Questions
mbti_questions = [
    "How do you prefer to spend your free time—socializing or enjoying time alone?",
    "When making decisions, do you rely more on logic or personal values?",
    "Do you find it easier to focus on details or the bigger picture?",
    "How comfortable are you with making decisions quickly without all the information?",
    "Do you prefer planning everything out or being spontaneous?",
    "How do you handle conflict—do you try to avoid it, or confront it directly?",
    "Do you rely more on past experiences or new data when making decisions?",
    "Do you find interacting with others energizing or draining?",
    "When working on a project, do you prefer to follow instructions or come up with your own approach?",
    "How often do you seek feedback from others when working on something?"
]

# =============== OCEAN MODEL CONFIGURATION ===============
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

# =============== HELPER FUNCTIONS ===============
def map_emotion_to_ocean(emotion):
    """Map detected emotion to OCEAN personality traits"""
    return emotion_to_ocean.get(emotion, emotion_to_ocean['neutral'])

def create_pie_chart(ocean_scores):
    """Create a pie chart of OCEAN personality traits"""
    labels = list(ocean_scores.keys())
    values = list(ocean_scores.values())
    
    # Expanded labels for readability
    label_map = {'O': 'Openness', 'C': 'Conscientiousness', 'E': 'Extraversion', 
                'A': 'Agreeableness', 'N': 'Neuroticism'}
    expanded_labels = [f"{label_map.get(k, k)}: {v:.2f}" for k, v in ocean_scores.items()]
    
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=expanded_labels, autopct='%1.1f%%', startangle=140)
    plt.title('OCEAN Personality Traits')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('static/ocean_pie_chart.png')
    plt.close()

def analyze_mbti_with_gemini(answers):
    """Use Gemini API to analyze MBTI personality type from questionnaire answers"""
    # Create a combined prompt with the answers
    prompt = f"""
    Based on the following answers to MBTI personality assessment questions, determine the most likely MBTI personality type.
    Please analyze each dimension separately and provide only the final 4-letter MBTI type (e.g., "INTJ").
    
    Questions and answers:
    {'-' * 50}
    """
    
    for i, (question, answer) in enumerate(zip(mbti_questions, answers)):
        prompt += f"\nQ{i+1}: {question}\nAnswer: {answer}\n"
    
    prompt += f"""
    {'-' * 50}
    
    Analyze each of the four dimensions:
    1. Introversion (I) vs. Extraversion (E)
    2. Intuition (N) vs. Sensing (S)
    3. Thinking (T) vs. Feeling (F)
    4. Judging (J) vs. Perceiving (P)
    
    Output only the 4-letter MBTI type in JSON format: {{"mbti_type": "XXXX"}}
    """
    
    # Configure the model
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Generate the response
    response = model.generate_content(prompt)
    
    try:
        # Try to parse the JSON response
        json_response = json.loads(response.text)
        mbti_type = json_response.get("mbti_type", "INFP")  # Default to INFP if parsing fails
    except:
        # If JSON parsing fails, try to extract just the MBTI type directly
        try:
            # Look for 4 letter combination matching MBTI format
            import re
            match = re.search(r'\b([EISNTFPJ]{4})\b', response.text)
            if match:
                mbti_type = match.group(1)
            else:
                mbti_type = "INFP"  # Default
        except:
            mbti_type = "INFP"  # Default as fallback
    
    return mbti_type

# =============== MODEL LOADING ===============
# Initialize global variables
aggregated_ocean_scores = []
current_mbti_answers = []

# Load OCEAN model
try:
    ocean_model = tf.keras.models.load_model('finalocean.h5')
    print("OCEAN model loaded successfully.")
except Exception as e:
    print(f"Error loading OCEAN model: {e}")
    print("Creating a new OCEAN model...")
    
    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_dim=6, activation='relu', name='input_layer'),  # Input: one-hot encoded emotions (6 categories)
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(5, activation='sigmoid')  # Output: OCEAN traits (5 traits)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Sample training data
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
    model.save('ocean_model.h5')
    ocean_model = model
    print("New OCEAN model trained and saved.")

# =============== ROUTES ===============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('mbti_questionnaire.html', questions=mbti_questions)

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global aggregated_ocean_scores
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi = frame[y:y+h, x:x+w]

                try:
                    analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    
                    dominant_emotion = analysis.get('dominant_emotion', None)
                    
                    if dominant_emotion:
                        # Map emotion to OCEAN traits
                        ocean_scores = map_emotion_to_ocean(dominant_emotion.lower())
                        aggregated_ocean_scores.append(ocean_scores)

                        # Display personality traits on frame
                        y_offset = 0
                        for trait, score in ocean_scores.items():
                            text = f"{trait}: {score:.2f}"
                            cv2.putText(frame, text, (x, y-10-y_offset), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
                            y_offset += 20

                except Exception as e:
                    print(f"Error analyzing face: {e}")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze', methods=['POST'])
def analyze_ocean():
    global aggregated_ocean_scores
    if aggregated_ocean_scores:
        # Calculate average scores
        final_scores = {}
        for trait in 'OCEAN':
            final_scores[trait] = np.mean([score.get(trait, 0.5) for score in aggregated_ocean_scores])
        
        create_pie_chart(final_scores)
        aggregated_ocean_scores.clear()  # Reset for next session
        return redirect(url_for('ocean_results'))
    return "No facial data captured. Please try again.", 400

@app.route('/submit_mbti', methods=['POST'])
def submit_mbti():
    answers = request.form.getlist('answers[]')
    
    # Use Gemini API to analyze MBTI type
    mbti_type = analyze_mbti_with_gemini(answers)
    
    return redirect(url_for('mbti_results', mbti_type=mbti_type))

@app.route('/results')
def ocean_results():
    return render_template('results.html', image_url='/static/ocean_pie_chart.png')

@app.route('/mbti_results/<mbti_type>')
def mbti_results(mbti_type):
    return render_template('mbti_results.html', mbti_type=mbti_type)

if __name__ == '__main__':
    app.run(debug=True)