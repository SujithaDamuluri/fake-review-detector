# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Step 1: Load the dataset
data_path = os.path.join('model', 'reviews.csv')
df = pd.read_csv(data_path)

# Step 2: Convert sentiment column to numeric label (0 = positive, 1 = negative)
df['label'] = df['sentiment'].apply(lambda x: 0 if x == 'positive' else 1)

# Step 3: Split data into train/test sets
X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create ML pipeline (TF-IDF + Logistic Regression)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Save the trained model to a file
joblib.dump(model, 'model/fake_review_model.pkl')

print("✅ Model training complete.")
print("📦 Model saved as: model/fake_review_model.pkl")
