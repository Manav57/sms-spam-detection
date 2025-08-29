from flask import Flask, render_template, request
import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        result = "🚨 Spam" if prediction == "spam" else "✅ Not Spam"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
