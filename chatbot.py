import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load files
with open('intents.json') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert to bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

# Predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append(classes[r[0]])

    return return_list

# Get response
def get_response(ints):
    tag = ints[0]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Chat loop
print("Chatbot is running! (type 'quit' to stop)")

while True:
    message = input("You: ")
    if message.lower() == "quit":
        break

    ints = predict_class(message)

    if ints:
        response = get_response(ints)
        print("Bot:", response)
    else:
        print("Bot: Sorry, I didn't understand that.")