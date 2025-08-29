import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ✅ File path
file_path = os.path.join(os.path.dirname(__file__), "SMSSpamCollection.txt")

# ✅ Ensure dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file not found at {file_path}. "
                            f"Make sure 'SMSSpamCollection.txt' is inside your project folder.")

# ✅ Load dataset
data = pd.read_csv(file_path, sep='\t', names=["label", "message"], encoding="latin-1")

# ✅ Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# ✅ Evaluate model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Test with custom messages
print("\n🔍 Testing with custom messages...")
test_messages = [
    "Congratulations! You won a $1000 gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT! Your account has been compromised, login immediately.",
    "Can you call me when you’re free?"
]

test_df = pd.DataFrame(test_messages, columns=['message'])
test_features = vectorizer.transform(test_df['message'])
predictions = model.predict(test_features)

for msg, label in zip(test_messages, predictions):
    print(f"Message: {msg} --> Prediction: {label}")

# ✅ Save model & vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n💾 Model and vectorizer saved as 'spam_model.pkl' and 'vectorizer.pkl'")
