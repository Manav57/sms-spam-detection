# 📧 Spam Email/SMS Detector

A machine learning project to classify SMS/Email messages as **Spam** or **Not Spam**.

---

## 🚀 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-detector.git
   cd spam-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (this generates `spam_model.pkl` and `vectorizer.pkl`):
   ```bash
   python "Spam Email Detection.py"
   ```

4. Run the web app:
   ```bash
   python app.py
   ```

5. Open browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

---

## 📂 Project Structure

```
spam-detector/
│── Spam Email Detection.py     # Training script (model building)
│── app.py                      # Flask web app
│── SMSSpamCollection.txt       # Dataset file (put it here)
│── spam_model.pkl              # Generated after training
│── vectorizer.pkl              # Generated after training
│── requirements.txt            # Dependencies
│── README.md                   # Documentation
│
├── templates/
│    └── index.html             # Web interface template
│
