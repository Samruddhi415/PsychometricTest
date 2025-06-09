import os
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from deepface import DeepFace
from transformers import TFBertModel, BertTokenizer
import re
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# MBTI Model Configuration
N_AXIS = 4
MAX_SEQ_LEN = 128
BERT_NAME = 'bert-base-uncased'
axes = ["I-E", "N-S", "T-F", "J-P"]
classes = {"I": 0, "E": 1,  # axis 1
          "N": 0, "S": 1,  # axis 2
          "T": 0, "F": 1,  # axis 3
          "J": 0, "P": 1}  # axis 4

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
    "How often do you seek feedback from others when working on something?",
    "Do you prefer a structured schedule or a more flexible approach to your day?",
    "Are you more comfortable expressing your emotions or keeping them to yourself?",
    "When faced with a problem, do you prefer to brainstorm ideas or research proven solutions?",
    "Do you tend to focus on what's happening right now or think ahead to the future?",
    "How do you usually approach challenges—by seeking guidance or trying to figure it out on your own?",
    "Do you feel more motivated by external rewards (recognition, praise) or internal satisfaction?",
    "How do you deal with new information—do you analyze it carefully or trust your gut instinct?",
    "Do you enjoy discussing abstract ideas or prefer to stick to practical topics?",
    "When working with a team, do you prefer taking the lead or being a supportive member?",
    "Do you tend to base your decisions more on logic or how they'll affect others emotionally?"
]

def map_emotion_to_ocean(emotion):
    """Map detected emotion to OCEAN personality traits"""
    # Basic mapping of emotions to OCEAN traits (simplified)
    mappings = {
        'happy': {'O': 0.7, 'C': 0.5, 'E': 0.8, 'A': 0.7, 'N': 0.3},
        'sad': {'O': 0.4, 'C': 0.4, 'E': 0.3, 'A': 0.5, 'N': 0.7},
        'angry': {'O': 0.3, 'C': 0.3, 'E': 0.5, 'A': 0.2, 'N': 0.8},
        'neutral': {'O': 0.5, 'C': 0.5, 'E': 0.5, 'A': 0.5, 'N': 0.5},
        'fear': {'O': 0.3, 'C': 0.4, 'E': 0.3, 'A': 0.4, 'N': 0.8},
        'surprise': {'O': 0.7, 'C': 0.4, 'E': 0.6, 'A': 0.6, 'N': 0.5},
        'disgust': {'O': 0.3, 'C': 0.5, 'E': 0.4, 'A': 0.3, 'N': 0.7}
    }
    return mappings.get(emotion, mappings['neutral'])

def text_preprocessing(text):
    """Preprocess text for BERT input"""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]
    return text

def prepare_bert_input(sentences, seq_len, bert_name):
    """Prepare input for BERT model"""
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    encodings = tokenizer(sentences, truncation=True, padding='max_length', max_length=seq_len)
    
    input = [np.array(encodings["input_ids"]),
             np.array(encodings["attention_mask"]),
             np.array(encodings.get("token_type_ids", np.zeros_like(encodings["input_ids"])))]
    
    return input

class CustomBERTModel(tf.keras.Model):
    def __init__(self, bert_name, num_classes):
        super(CustomBERTModel, self).__init__()
        self.bert = TFBertModel.from_pretrained(bert_name)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.pooling(bert_outputs.last_hidden_state)
        return self.classifier(pooled_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "bert_name": self.bert.name,
            "num_classes": self.classifier.units
        })
        return config

# Register the custom object
tf.keras.utils.get_custom_objects()["CustomBERTModel"] = CustomBERTModel

def create_pie_chart(ocean_scores):
    """Create a pie chart of OCEAN personality traits"""
    plt.figure(figsize=(8, 8))
    plt.pie(
        list(ocean_scores.values()), 
        labels=[f'{trait}\n{score:.2f}' for trait, score in ocean_scores.items()],
        autopct='%1.1f%%'
    )
    plt.title('OCEAN Personality Traits')
    plt.tight_layout()
    plt.savefig('static/ocean_pie_chart.png')
    plt.close()

# Load or create MBTI model
try:
    mbti_model = tf.keras.models.load_model('mbti_model.h5', 
                                          custom_objects={'CustomBERTModel': CustomBERTModel})
    print("MBTI model loaded successfully.")
except Exception as e:
    print(f"Error loading MBTI model: {e}")
    print("Creating a new MBTI model...")
    
    # Create and compile the model if not found
    mbti_model = CustomBERTModel(BERT_NAME, N_AXIS)
    input_ids = tf.keras.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
    token_type_ids = tf.keras.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
    
    output = mbti_model([input_ids, attention_mask, token_type_ids])
    mbti_model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)
    
    opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = keras.losses.BinaryCrossentropy()
    mbti_model.compile(loss=loss, optimizer=opt, metrics=[keras.metrics.AUC(multi_label=True, curve="ROC")])
    
    # Sample training data (replace with real data)
    sample_texts = [
        "I enjoy spending time alone and thinking deeply about life.",
        "I prefer to make decisions based on logic rather than emotions.",
        "I find it easier to focus on details rather than the bigger picture.",
        "I am comfortable making quick decisions without all the information.",
        "I prefer planning everything out rather than being spontaneous.",
        "I try to avoid conflict rather than confront it directly.",
        "I rely more on past experiences than new data when making decisions.",
        "I find interacting with others energizing rather than draining.",
        "I prefer to follow instructions rather than come up with my own approach.",
        "I often seek feedback from others when working on something.",
        "I prefer a structured schedule rather than a flexible approach to my day.",
        "I am more comfortable expressing my emotions rather than keeping them to myself.",
        "I prefer to brainstorm ideas rather than research proven solutions when faced with a problem.",
        "I tend to focus on what's happening right now rather than think ahead to the future.",
        "I usually approach challenges by seeking guidance rather than trying to figure it out on my own.",
        "I feel more motivated by external rewards rather than internal satisfaction.",
        "I analyze new information carefully rather than trusting my gut instinct.",
        "I enjoy discussing abstract ideas rather than sticking to practical topics.",
        "I prefer taking the lead rather than being a supportive member when working with a team.",
        "I tend to base my decisions more on logic rather than how they'll affect others emotionally."
    ]
    sample_labels = np.array([
        [0, 0, 0, 0],  # I, N, T, J
        [1, 0, 0, 0],  # E, N, T, J
        [0, 1, 0, 0],  # I, S, T, J
        [1, 1, 0, 0],  # E, S, T, J
        [0, 0, 1, 0],  # I, N, F, J
        [1, 0, 1, 0],  # E, N, F, J
        [0, 1, 1, 0],  # I, S, F, J
        [1, 1, 1, 0],  # E, S, F, J
        [0, 0, 0, 1],  # I, N, T, P
        [1, 0, 0, 1],  # E, N, T, P
        [0, 1, 0, 1],  # I, S, T, P
        [1, 1, 0, 1],  # E, S, T, P
        [0, 0, 1, 1],  # I, N, F, P
        [1, 0, 1, 1],  # E, N, F, P
        [0, 1, 1, 1],  # I, S, F, P
        [1, 1, 1, 1],  # E, S, F, P
        [0, 0, 0, 0],  # I, N, T, J
        [1, 0, 0, 0],  # E, N, T, J
        [0, 1, 0, 0],  # I, S, T, J
        [1, 1, 0, 0]   # E, S, T, J
    ])
    
    # Preprocess sample texts
    preprocessed_texts = [text_preprocessing(text) for text in sample_texts]
    bert_input = prepare_bert_input(preprocessed_texts, MAX_SEQ_LEN, BERT_NAME)
    
    # Train the model
    mbti_model.fit(bert_input, sample_labels, epochs=10, batch_size=8, validation_split=0.2)
    
    # Save the newly trained model
    mbti_model.save('mbti_model.h5')
    print("New MBTI model trained and saved.")

# Initialize global variables
aggregated_ocean_scores = []
current_mbti_answers = []

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
                        ocean_scores = map_emotion_to_ocean(dominant_emotion.lower())
                        aggregated_ocean_scores.append(ocean_scores)

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
        final_scores = {trait: np.mean([score[trait] for score in aggregated_ocean_scores]) 
                       for trait in aggregated_ocean_scores[0].keys()}
        
        create_pie_chart(final_scores)
        aggregated_ocean_scores.clear()
        return redirect(url_for('ocean_results'))
    return "No data captured. Please try again.", 400

@app.route('/submit_mbti', methods=['POST'])
def submit_mbti():
    global current_mbti_answers
    answers = request.form.getlist('answers[]')
    current_mbti_answers = answers
    
    # Prepare text input for BERT
    preprocessed_text = [text_preprocessing(answer) for answer in answers]
    bert_input = prepare_bert_input(preprocessed_text, MAX_SEQ_LEN, BERT_NAME)
    
    # Get model predictions
    predictions = mbti_model.predict(bert_input)
    
    # Calculate MBTI type
    mbti_type = ""
    for i, (axis, pred) in enumerate(zip(axes, predictions.mean(axis=0))):
        mbti_type += axis.split('-')[0] if pred < 0.5 else axis.split('-')[1]
    
    return redirect(url_for('mbti_results', mbti_type=mbti_type))

@app.route('/results')
def ocean_results():
    return render_template('results.html', image_url='/static/ocean_pie_chart.png')

@app.route('/mbti_results/<mbti_type>')
def mbti_results(mbti_type):
    return render_template('mbti_results.html', mbti_type=mbti_type)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True)