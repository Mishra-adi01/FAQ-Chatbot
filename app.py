from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import csv
import string
from difflib import SequenceMatcher

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load FAQ data
faq_data = []

with open("faq_data.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        faq_data.append({
            "question": row["Question"].lower(),
            "answer": row["Answer"]
        })

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Find best match using fuzzy matching
def find_best_match(user_question):
    user_q = preprocess(user_question)
    best_match = None
    best_score = 0
    
    for faq in faq_data:
        faq_q = preprocess(faq["question"])
        
        # Calculate similarity
        score = SequenceMatcher(None, user_q, faq_q).ratio()
        
        # Check if any word matches
        user_words = set(user_q.split())
        faq_words = set(faq_q.split())
        common_words = user_words.intersection(faq_words)
        
        # Boost score if common words exist
        if common_words:
            score += len(common_words) * 0.1
        
        if score > best_score:
            best_score = score
            best_match = faq
    
    return best_match, best_score

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"response": "Please ask a question."})

        # Find best matching FAQ
        best_match, score = find_best_match(user_message)

        # Debug output
        print(f"User: {user_message}")
        if best_match:
            print(f"Best match: {best_match['question']}")
            print(f"Score: {score}")

        # Return answer if good match found
        if best_match and score > 0.3:
            reply = best_match["answer"]
        else:
            reply = "Sorry, I couldn't find a relevant answer. Please try rephrasing your question."

        return jsonify({"response": reply})

    except Exception as e:
        print("Error:", e)
        return jsonify({"response": "Server error."}), 500

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
