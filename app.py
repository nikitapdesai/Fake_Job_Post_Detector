from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Same cleaning function used during training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        user_input = request.form["jobtext"]

        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        if pred == 1:
            prediction = "Fake Job Posting Detected"
        else:
            prediction = "Legitimate Job Posting"

        probability = f"{prob*100:.2f}%"

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)