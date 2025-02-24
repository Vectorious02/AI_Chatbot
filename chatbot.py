import os
import requests
from dotenv import load_dotenv # type: ignore
import numpy as np
import nltk
import json
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model # type: ignore

# Load environment variables
load_dotenv()

# Initialize components
lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')

with open('intents.json') as file:
    data = json.load(file)

words = data['words']
classes = data['classes']

# ----------------- API Functions -----------------
def get_weather(user_input):
    api_key = os.getenv('OPENWEATHER_API_KEY')
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    try:
        # List of supported Indian cities
        indian_cities = {
            'delhi', 'kolkata', 'mumbai','chennai', 'bangalore',
            'hyderabad', 'pune', 'noida', 'ghaziabad', 'jaipur',
            'gurgaon', 'ahmedabad', 'lucknow', 'kanpur', 'nagpur' 
        }
        
        # City name synonyms
        city_synonyms = {
            'new delhi': 'Delhi',
            'bengaluru': 'Bangalore',
            'calcutta': 'Kolkata',
            'gurgaon': 'Gurugram'
        }
        
        # Extract city name from input
        words = nltk.word_tokenize(user_input.lower())
        detected_cities = [word.title() for word in words if word in indian_cities]
        
        # Fallback to POS tagging if no city detected
        if not detected_cities:
            pos_tags = nltk.pos_tag(nltk.word_tokenize(user_input))
            detected_cities = [word for word, pos in pos_tags if pos == 'NNP']
        
        # Handle no city detected
        if not detected_cities:
            return "Please specify an Indian city (e.g. 'weather in Delhi')."
        
        # Use the first detected city
        city = detected_cities[0]
        
        # Replace with synonym if needed
        city = city_synonyms.get(city.lower(), city)
        
        # API call
        params = {
            'q': city + ',IN',  # Add country code for India
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(base_url, params=params).json()
        
        # Error handling
        if response.get('cod') != 200:
            return f"Couldn't fetch weather for {city}. Try another city!"
        
        # Extract weather data
        weather_desc = response['weather'][0]['description'].capitalize()
        temp = response['main']['temp']
        humidity = response['main']['humidity']
        
        return (f"Weather in {city}:\n"
                f"🌡 Temperature: {temp}°C\n"
                f"☁️ Condition: {weather_desc}\n"
                f"💧 Humidity: {humidity}%")
                
    except Exception as e:
        return f"Weather service error: {str(e)}"

def get_news(user_input):
    api_key = os.getenv('NEWS_API_KEY')
    base_url = "https://newsapi.org/v2/top-headlines"
    
    try:
        # Detect category from user input
        categories = {
            'technology': ['tech', 'technology', 'gadgets'],
            'business': ['business', 'finance', 'market'],
            'sports': ['sports', 'cricket', 'football'],
            'entertainment': ['entertainment', 'movies', 'celebrity']
        }
        
        detected_category = 'general'
        for category, keywords in categories.items():
            if any(word in user_input.lower() for word in keywords):
                detected_category = category
                break

        params = {
            'apiKey': api_key,
            'category': detected_category,
            'pageSize': 3,  # Get 3 articles
            'language': 'en'
        }

        response = requests.get(base_url, params=params).json()
        
        if response['status'] != 'ok':
            error_message = response.get('message', 'Unknown error')
            return f"News error: {error_message}"
            
        articles = response['articles']
        if not articles:
            return "No recent news found in this category."
            
        news_list = []
        for idx, article in enumerate(articles[:3], 1):
            title = article['title'].split(' - ')[0]  # Remove source from title
            news_list.append(f"{idx}. {title}")
            
        return f"📰 Top {detected_category} news:\n" + "\n".join(news_list)
        
    except Exception as e:
        return f"News service unavailable: {str(e)}"

# ----------------- Chatbot Logic -----------------
def preprocess_input(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    bag = [0] * len(words)
    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1
    return np.array([bag])

def get_response(user_input):
    processed_input = preprocess_input(user_input)
    predictions = model.predict(processed_input)[0]
    predicted_index = np.argmax(predictions)
    tag = classes[predicted_index]
    
    # API Integrations
    if tag == "weather":
        cities = [word.title() for word in nltk.word_tokenize(user_input) if word.title() in ["London", "Paris", "Mumbai", "Delhi","Kolkata","Noida","Gurgaon","Chennai", "Bangalore",
    "Hyderabad", "Pune", "Ghaziabad", "Jaipur",
     "Ahmedabad", "Lucknow", "Kanpur", "Nagpur"]]
        city = cities[0] if cities else "Mumbai"  # Default city
        return get_weather(city)
        
    elif tag == "news":
        categories = ['technology', 'business', 'sports']
        detected_category = next((word for word in nltk.word_tokenize(user_input) if word in categories), 'general')
        return get_news(detected_category)
        
    else:
        for intent in data['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    
    return "I'm still learning! Could you rephrase that?"

# ----------------- Chat Interface -----------------
print("Chatbot: Namaskar Mitro, I can now provide real-time weather in India and top news as well as much more. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = get_response(user_input)
    print(f"Chatbot: {response}")