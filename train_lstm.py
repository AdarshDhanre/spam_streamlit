import os
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# =============================
# PATH CONFIG
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "spam.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "dl_model")

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# LOAD DATASET (KAGGLE FORMAT)
# =============================
df = pd.read_csv(DATA_PATH, encoding="latin-1")

# Kaggle spam.csv uses v1 (label), v2 (message)
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# =============================
# TEXT CLEANING
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
# TOKENIZER
# =============================
tokenizer = Tokenizer(
    num_words=5000,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(
    X_seq,
    maxlen=100,
    padding="post",
    truncating="post"
)

# =============================
# TRAIN–TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# LSTM MODEL (STABLE)
# =============================
model = Sequential()
model.add(Embedding(
    input_dim=5000,
    output_dim=64,
    input_length=100
))
model.add(LSTM(64))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =============================
# TRAIN MODEL
# =============================
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# =============================
# SAVE MODEL (MODERN FORMAT)
# =============================
model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ LSTM model trained and saved successfully!")
