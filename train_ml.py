import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =============================
# PATHS
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "spam.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "ml_model")

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# LOAD DATASET (KAGGLE)
# =============================
df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# =============================
# CLEAN TEXT
# =============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['message'] = df['message'].apply(clean_text)

X = df['message']
y = df['label']

# =============================
# TF-IDF
# =============================
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# =============================
# TRAIN
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =============================
# EVALUATE
# =============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ ML Model Accuracy: {acc:.4f}")

# =============================
# SAVE
# =============================
pickle.dump(model, open(os.path.join(MODEL_DIR, "ml_model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb"))

print("✅ ML model & vectorizer saved successfully!")
