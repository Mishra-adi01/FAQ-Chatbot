import random
import json
import pickle
import numpy as np
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

# -------------------------------
# KEYWORD RULE SYSTEM
# -------------------------------

keyword_map = {
    "internship_duration": ["duration", "months", "how long"],
    "eligibility_criteria": ["eligible", "eligibility", "apply"],
    "required_skills": ["skills", "technical", "requirements"],
    "selection_process": ["selection", "interview", "hiring", "round"],
    "stipend_details": ["stipend", "paid", "salary", "earn"],
    "certificate_policy": ["certificate"],
    "internship_mode": ["remote", "online", "onsite", "mode"],
    "contact_support": ["contact", "hr", "email", "number", "support"],
    "placement_opportunity": ["placement", "job opportunity"],
    "mock_interview_support": ["mock", "interview preparation"]
}

def keyword_match(sentence):
    sentence = sentence.lower()
    for tag, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in sentence:
                return tag
    return None

# -------------------------------
# TEXT PREPROCESSING
# -------------------------------

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

# -------------------------------
# MODEL PREDICTION
# -------------------------------

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]

    ERROR_THRESHOLD = 0.60  # Increased for better accuracy

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        return classes[results[0][0]]  # Return highest confidence tag
    else:
        return None

# -------------------------------
# GET RESPONSE
# -------------------------------

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand."

# -------------------------------
# CHAT LOOP
# -------------------------------

print("Chatbot is running! (type 'quit' to stop)")

while True:
    message = input("You: ")

    if message.lower() == "quit":
        print("Bot: Goodbye! 👋")
        break

    # 1️⃣ Keyword check first
    keyword_tag = keyword_match(message)

    if keyword_tag:
        response = get_response(keyword_tag)
        print("Bot:", response)
        continue

    # 2️⃣ ML prediction
    predicted_tag = predict_class(message)

    if predicted_tag:
        response = get_response(predicted_tag)
        print("Bot:", response)
    else:
        # 3️⃣ Fallback
        fallback_response = get_response("fallback")
        print("Bot:", fallback_response)