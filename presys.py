from flask import Flask, request, jsonify, render_template
import pandas as pd
import mysql.connector

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

try:

    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Teju@8978",
        database="chatbot_system"
    )

    cursor = db.cursor()

    print("MySQL Connected Successfully")

except Exception as e:

    print("Database Error:", e)

try:

    data = pd.read_csv("sys.csv")

    data = data.fillna("")

    print("Dataset Loaded Successfully")

except Exception as e:

    print("CSV Error:", e)

questions = data['question']
answers = data['answer']

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(questions)

def detect_emotion(text):

    emotion_words = [
        "stress",
        "depressed",
        "worried"
    ]

    for word in emotion_words:

        if word in text.lower():

            return """
Don't feel alone.<br>
Talk with your friends and family.<br>
Spend time with trusted people.<br>
Take care of yourself.
"""

    return None

def predict_answer(question):

    emotion_response = detect_emotion(question)

    if emotion_response:

        return emotion_response, [], "Emotional Support", 100

    user_vector = vectorizer.transform([question])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_index = similarity.argmax()

    confidence = round(
        similarity[0][best_index] * 100,
        2
    )

    answer = answers.iloc[best_index]

    options = []

    if 'option1' in data.columns:
        options.append(data.iloc[best_index].get('option1', ''))

    if 'option2' in data.columns:
        options.append(data.iloc[best_index].get('option2', ''))

    if 'option3' in data.columns:
        options.append(data.iloc[best_index].get('option3', ''))

    options = [opt for opt in options if opt != ""]

    return answer, options, "Normal", confidence

@app.route('/')
def home():

    return render_template('pre.html')

@app.route('/ask', methods=['POST'])
def ask():

    try:

        data_json = request.get_json()

        if not data_json:

            return jsonify({
                "error": "No JSON data received"
            }), 400

        user_question = data_json.get('question')

        if not user_question:

            return jsonify({
                "error": "Question is required"
            }), 400

        answer, options, emotion, confidence = predict_answer(
            user_question
        )

        query = """
        INSERT INTO chat_logs
        (user_question, predicted_answer, emotion, accuracy)
        VALUES (%s, %s, %s, %s)
        """

        values = (
            user_question,
            answer,
            emotion,
            confidence
        )

        cursor.execute(query, values)

        db.commit()

        return jsonify({
            "question": user_question,
            "answer": answer,
            "recommended_options": options,
            "emotion_type": emotion,
            "accuracy_score": confidence
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

@app.route('/feedback', methods=['POST'])
def feedback():

    try:

        data_json = request.get_json()

        feedback_text = data_json.get('feedback', '')

        response = f"""
Sorry you are not satisfied.<br><br>

Your feedback:<br>
{feedback_text}<br><br>

Please explain your issue again.<br>
We will recommend another solution.
"""

        return jsonify({
            "message": response
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':

    app.run(debug=True)
