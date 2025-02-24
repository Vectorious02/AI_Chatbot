from flask import Flask, render_template, request, jsonify
import numpy as np
import nltk
import json
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model # type: ignore

from chatbot import get_response # type: ignore

app = Flask(__name__)
model = load_model('model.h5')

# Load intents and preprocessing data
with open('intents.json') as file:
    data = json.load(file)
words = data['words']
classes = data['classes']
lemmatizer = WordNetLemmatizer()

def preprocess_input(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    bag = [0] * len(words)
    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1
    return np.array([bag])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']
    response_text = get_response(user_input)  # Ensure this is a string
    return jsonify({'response': str(response_text)})  # Convert to string 
    tag = classes[np.argmax(pred)]
    for intent in data['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return jsonify({'response': response})
    return jsonify({'response': "I didn't understand that."})

if __name__ == '__main__':
    app.run(debug=True)