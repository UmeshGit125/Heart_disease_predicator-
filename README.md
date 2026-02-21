# ❤️ Heart Disease Predictor (KNN Model)

A Machine Learning web application that predicts the likelihood of heart disease using patient medical data.

This project uses the K-Nearest Neighbors (KNN) classification algorithm and is deployed using Streamlit.

---

## 📌 Project Overview

Heart Disease Predictor is a supervised machine learning classification project that predicts whether a patient is at risk of heart disease based on clinical features.

The model is trained on a structured medical dataset and deployed through a user-friendly Streamlit interface.

---

## 🧠 Machine Learning Approach

- Type: Supervised Learning (Classification)
- Algorithm Used: K-Nearest Neighbors (KNN)
- Distance Metric: Euclidean Distance
- Evaluation Metrics:
  - Accuracy Score
  - Confusion Matrix

KNN works by identifying the 'K' nearest data points to a new input sample and assigning the majority class among those neighbors.

---

## 📊 Features Used

The model uses the following medical parameters:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- ST Depression
- Slope of Peak Exercise ST Segment
- Number of Major Vessels
- Thalassemia

---

## 🔄 Project Workflow

1. Data Cleaning & Preprocessing
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling (important for KNN)

2. Train-Test Split

3. Model Training using KNN

4. Model Evaluation

5. Deployment using Streamlit

---

## 🚀 Installation & Setup (Using UV)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/UmeshGit125/Heart_disease_predicator-.git
cd Heart_disease_predicator-
```
### 2️⃣ Create Virtual Environment using UV
    uv venv
Activate the envoirment

.venv\Scripts\activate

### 3️⃣ Install Dependencies
  uv pip install -r requirements.txt 

### ▶️ Run the Application
      streamlit run app.py 
      http://localhost:8501


## 👨‍💻 Author

Umesh
Machine Learning Enthusiast
