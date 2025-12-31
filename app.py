import os
import re
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from auth import create_user_table, add_user, login_user

# -----------------------------
# INIT DB
# -----------------------------
create_user_table()

# -----------------------------
# SESSION STATE
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if "refresh" not in st.session_state:
    st.session_state.refresh = 0  # counter to trigger rerun logic

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

ML_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml_model", "ml_model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "ml_model", "vectorizer.pkl")
DL_MODEL_PATH = os.path.join(PROJECT_ROOT, "dl_model", "lstm_model.keras")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "dl_model", "tokenizer.pkl")

# -----------------------------
# LOAD MODELS
# -----------------------------
ml_model = vectorizer = dl_model = tokenizer = None

if os.path.exists(ML_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(ML_MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
else:
    st.warning("ML model or vectorizer not found. ML prediction disabled.")

if os.path.exists(DL_MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    dl_model = load_model(DL_MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
else:
    st.info("DL model or tokenizer not found. DL prediction disabled.")

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# LOGIN / SIGNUP PAGE
# -----------------------------
def login_page():
    st.title("🔐 Login / Signup")
    choice = st.selectbox("Choose", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Signup" and st.button("Create Account"):
        if add_user(username, password):
            st.success("✅ Account created. Please login.")
            st.session_state.refresh += 1
        else:
            st.error("❌ Username already exists")
    
    elif choice == "Login" and st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username} 🎉")
            st.session_state.refresh += 1
        else:
            st.error("❌ Invalid username or password")

# -----------------------------
# MAIN APP
# -----------------------------
def main_app():
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.refresh += 1

    st.title("📩 SMS Spam Detection System")
    msg = st.text_area("Enter SMS message")

    if st.button("Predict"):
        if msg.strip() == "":
            st.warning("⚠ Please enter a message")
            return
        
        cleaned = clean_text(msg)

        # ML Prediction
        if ml_model and vectorizer:
            try:
                ml_pred = ml_model.predict(vectorizer.transform([cleaned]))[0]
                st.subheader("Machine Learning Prediction:")
                if ml_pred == 1:
                    st.error("🚨 SPAM MESSAGE")
                else:
                    st.success("✅ NOT SPAM")
            except Exception as e:
                st.error(f"ML Prediction error: {e}")
        else:
            st.info("ML model not loaded. Skipping ML prediction.")

       
# -----------------------------
# ROUTING
# -----------------------------
# Use a dummy button to trigger a rerun whenever refresh counter changes
st.session_state.refresh  # access it to trigger rerun

if st.session_state.logged_in:
    main_app()
else:
    login_page()
