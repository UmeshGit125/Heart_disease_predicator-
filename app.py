import streamlit as st
import pandas as pd
import joblib
import os

USER_FILE = "users.csv"

if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["username", "password"]).to_csv(USER_FILE, index=False)

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

def load_users():
    return pd.read_csv(USER_FILE)

def register_user(username, password):
    username = username.strip()
    password = password.strip()
    users = load_users()
    if username in users["username"].astype(str).str.strip().values:
        return False
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_FILE, index=False)
    return True

for key in ["logged_in", "username", "show_register", "dark_mode"]:
    if key not in st.session_state:
        st.session_state[key] = False if key != "username" else ""

def apply_theme():
    if st.session_state.dark_mode:
        st.markdown("""
            <style>
            .stApp {
                background-color: #0E1117 !important;
                color: #FAFAFA !important;
            }
            html, body, [class*="css"] {
                background-color: #0E1117 !important;
                color: #FAFAFA !important;
            }
            h1, h2, h3, h4, h5, h6,
            p, label, span, div, section, input, textarea {
                color: #E0E0E0 !important;
            }
            input, textarea, .stNumberInput input, .stTextInput input {
                background-color: #20252B !important;
                color: white !important;
                border: 1px solid #5A5A5A !important;
            }
            .stSelectbox > div {
                background-color: #20252B !important;
                color: white !important;
            }
            div[data-baseweb="popover"] {
                background-color: white !important;
                color: black !important;
                border-radius: 6px;
                border: 1px solid #AAA;
            }
            div[data-baseweb="popover"] > div > div {
                background-color: white !important;
                color: black !important;
            }
            .stButton > button {
                background-color: #20252B !important;
                color: #FAFAFA !important;
                border: 1px solid #5A5A5A !important;
                border-radius: 6px;
                transition: 0.3s;
            }
            .stButton > button:hover {
                background-color: #323840 !important;
                color: #FFFFFF !important;
            }
            ::placeholder {
                color: #A0A0A0 !important;
            }
            .stSlider > div[data-baseweb="slider"] {
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)

def show_login():
    apply_theme()
    st.markdown("### <span style='color:#00FFAA'>Welcome to HeartMate 💓</span>", unsafe_allow_html=True)
    st.markdown("#### <span style='color:#CCCCCC'>Login to check your heart stroke risk</span>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()
        username = username.strip()
        password = password.strip()
        user_match = users[
            (users["username"].astype(str).str.strip() == username) &
            (users["password"].astype(str).str.strip() == password)
        ]
        if not user_match.empty:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.info("Don't have an account?")
    if st.button("Go to Register"):
        st.session_state.show_register = True
        st.rerun()

def show_register():
    apply_theme()
    st.markdown("### <span style='color:#FFD700'>Create an Account</span>", unsafe_allow_html=True)

    new_user = st.text_input("Choose a Username")
    new_pass = st.text_input("Choose a Password", type="password")

    if st.button("Register"):
        if new_user == "" or new_pass == "":
            st.warning("⚠️ Please fill all fields.")
        elif register_user(new_user, new_pass):
            st.success("🎉 Registered successfully! Now login.")
            st.session_state.show_register = False
            st.rerun()
        else:
            st.error("❗ Username already exists.")

    if st.button("Go to Login"):
        st.session_state.show_register = False
        st.rerun()

def show_prediction():
    apply_theme()
    st.title("💓 Heart Stroke Prediction")
    st.markdown(f"👤 Logged in as: **{st.session_state.username}**")

    st.markdown("### ⚙️ Settings")
    st.checkbox("🌙 Enable Dark Mode", key="dark_mode", on_change=st.rerun)

    st.markdown("### 🔎 Enter Your Health Data:")

    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    if st.button("🧠 Predict Risk"):
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            f'Sex_{sex}': 1,
            f'ChestPainType_{chest_pain}': 1,
            f'RestingECG_{resting_ecg}': 1,
            f'ExerciseAngina_{exercise_angina}': 1,
            f'ST_Slope_{st_slope}': 1
        }

        input_df = pd.DataFrame([raw_input])
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    if st.button("🔓 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

if st.session_state.logged_in:
    show_prediction()
elif st.session_state.show_register:
    show_register()
else:
    show_login()
