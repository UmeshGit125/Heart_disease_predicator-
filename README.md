HeartMate is a user-friendly web application developed using Streamlit that allows users to predict their risk of heart stroke based on medical input parameters. It includes a secure login/registration system and offers a clean, modern UI with optional dark mode support.

This application uses a pre-trained K-Nearest Neighbors (KNN) machine learning model to analyze health-related inputs and deliver a risk prediction. It is ideal for demonstrating applied ML, Streamlit-based UI development, and user authentication concepts.

🔑 Key Features
🔐 Login/Register: New users can register and securely log in (credentials are stored in users.csv).

💡 Health Prediction: Predicts heart disease risk based on user input.

🌙 Dark Mode: Toggle between light and dark themes for better accessibility.

🎯 Real-time Feedback: Instant prediction result after form submission.

🛠 Technologies Used
Frontend/UI: Streamlit

ML Model: KNN Classifier (trained externally and loaded via joblib)

Data Processing: pandas, scikit-learn StandardScaler

Storage: CSV file for storing registered users

📦 Project Structure
bash
Copy
Edit
project/
│
├── app.py                # Main Streamlit app
├── knn_heart_model.pkl   # Trained KNN model
├── heart_scaler.pkl      # StandardScaler object
├── heart_columns.pkl     # Required feature columns
├── users.csv             # User credentials (created at runtime)
└── README.md             # Project documentation

-- the ml used algo is added in repo
