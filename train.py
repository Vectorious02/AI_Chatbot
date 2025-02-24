import numpy as np
import nltk
import json
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore

nltk.download('punkt')
nltk.download('wordnet')

# Load intents.json
with open('intents.json') as file:
    data = json.load(file)

# Preprocess data
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize words
        word_list = nltk.word_tokenize(pattern.lower())
        words.extend([lemmatizer.lemmatize(word) for word in word_list if word not in ignore_chars])
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Remove duplicates and sort
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Create training data (bag-of-words)
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and split data
random.shuffle(training)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile and train
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('model.h5')

# Save words and classes to intents.json
data['words'] = words
data['classes'] = classes
with open('intents.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Model trained and saved!")