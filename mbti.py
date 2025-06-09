import os
from flask import Flask, render_template, Response, request, redirect, url_for
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import re
import matplotlib.pyplot as plt

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
    
    input_ids = np.array(encodings["input_ids"])
    attention_mask = np.array(encodings["attention_mask"])
    token_type_ids = np.array(encodings.get("token_type_ids", np.zeros_like(encodings["input_ids"])))
    
    return [input_ids, attention_mask, token_type_ids]

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

# Create a simple rule-based fallback model in case the BERT model can't be loaded
def predict_mbti_rule_based(answers):
    """Simple rule-based model for MBTI prediction"""
    ie_score = 0
    ns_score = 0
    tf_score = 0
    jp_score = 0
    
    # Keywords associated with each dichotomy
    ie_keywords = {
        'introvert': -1, 'introversion': -1, 'alone': -1, 'quiet': -1, 'internal': -1, 'reflect': -1, 'private': -1,
        'extrovert': 1, 'extraversion': 1, 'people': 1, 'social': 1, 'talkative': 1, 'outgoing': 1, 'energizing': 1
    }
    
    ns_keywords = {
        'intuition': -1, 'abstract': -1, 'big picture': -1, 'concept': -1, 'pattern': -1, 'meaning': -1, 'possibility': -1,
        'sensing': 1, 'concrete': 1, 'detail': 1, 'fact': 1, 'realistic': 1, 'practical': 1, 'specific': 1
    }
    
    tf_keywords = {
        'thinking': -1, 'logic': -1, 'analytical': -1, 'objective': -1, 'rational': -1, 'fair': -1, 'principle': -1,
        'feeling': 1, 'emotion': 1, 'value': 1, 'empathy': 1, 'harmony': 1, 'consensus': 1, 'personal': 1
    }
    
    jp_keywords = {
        'judging': -1, 'plan': -1, 'structure': -1, 'organized': -1, 'decide': -1, 'control': -1, 'schedule': -1,
        'perceiving': 1, 'spontaneous': 1, 'flexible': 1, 'adapt': 1, 'explore': 1, 'open-ended': 1, 'flow': 1
    }
    
    # Process each answer
    for answer in answers:
        answer = answer.lower()
        
        # Check IE dichotomy
        for keyword, value in ie_keywords.items():
            if keyword in answer:
                ie_score += value
        
        # Check NS dichotomy
        for keyword, value in ns_keywords.items():
            if keyword in answer:
                ns_score += value
        
        # Check TF dichotomy
        for keyword, value in tf_keywords.items():
            if keyword in answer:
                tf_score += value
        
        # Check JP dichotomy
        for keyword, value in jp_keywords.items():
            if keyword in answer:
                jp_score += value
    
    # Determine type
    mbti_type = ""
    mbti_type += "I" if ie_score <= 0 else "E"
    mbti_type += "N" if ns_score <= 0 else "S"
    mbti_type += "T" if tf_score <= 0 else "F"
    mbti_type += "J" if jp_score <= 0 else "P"
    
    return mbti_type

# Try to load the model, or create a simpler one if loading fails
try:
    mbti_model = tf.keras.models.load_model('mbti_model.h5', 
                                         custom_objects={'CustomBERTModel': CustomBERTModel})
    print("MBTI model loaded successfully.")
    model_available = True
except Exception as e:
    print(f"Error loading MBTI model: {e}")
    print("Will use rule-based fallback model for prediction.")
    model_available = False

# Initialize global variable
current_mbti_answers = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('mbti_questionnaire.html', questions=mbti_questions)

@app.route('/submit_mbti', methods=['POST'])
def submit_mbti():
    global current_mbti_answers
    answers = request.form.getlist('answers[]')
    current_mbti_answers = answers
    
    try:
        if model_available:
            # Prepare text input for BERT
            preprocessed_text = [text_preprocessing(answer) for answer in answers]
            bert_input = prepare_bert_input(preprocessed_text, MAX_SEQ_LEN, BERT_NAME)
            
            # Get model predictions
            predictions = mbti_model.predict(bert_input)
            
            # Calculate MBTI type based on average predictions
            avg_preds = predictions.mean(axis=0)
            mbti_type = ""
            for i, (axis, pred) in enumerate(zip(axes, avg_preds)):
                mbti_type += axis.split('-')[0] if pred < 0.5 else axis.split('-')[1]
        else:
            # Use rule-based fallback
            mbti_type = predict_mbti_rule_based(answers)
            
    except Exception as e:
        print(f"Error making MBTI prediction: {e}")
        # Use rule-based fallback on error
        mbti_type = predict_mbti_rule_based(answers)
    
    return redirect(url_for('mbti_results', mbti_type=mbti_type))

@app.route('/mbti_results/<mbti_type>')
def mbti_results(mbti_type):
    return render_template('mbti_results.html', mbti_type=mbti_type)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)