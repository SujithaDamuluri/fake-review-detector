# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the app
app = Flask(__name__)

# Load trained model
model = joblib.load('model/fake_review_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']

        # Make prediction
        pred = model.predict([review])[0]
        prob = model.predict_proba([review])[0]

        # Interpret result
        label = "🟢 Genuine Review" if pred == 0 else "🔴 Fake Review"
        confidence = round(np.max(prob) * 100, 2)

        return render_template('index.html', prediction=label, confidence=confidence)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
