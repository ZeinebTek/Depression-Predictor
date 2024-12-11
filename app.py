from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'assets/best_model.joblib')

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load the encoders and scaler used during training
feature_encoders_path = os.path.join(os.path.dirname(__file__), 'assets/feature_encoders.joblib')
scaler_path = os.path.join(os.path.dirname(__file__), 'assets/scaler.joblib')
target_encoder_path = os.path.join(os.path.dirname(__file__), 'assets/target_encoder.joblib')

# Check if the encoder and scaler files exist
if not os.path.exists(feature_encoders_path):
    raise FileNotFoundError(f"Feature encoders file not found: {feature_encoders_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
if not os.path.exists(target_encoder_path):
    raise FileNotFoundError(f"Target encoder file not found: {target_encoder_path}")

feature_encoders = joblib.load(feature_encoders_path)
scaler = joblib.load(scaler_path)
target_encoder = joblib.load(target_encoder_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        gender = request.form['gender']
        age = int(request.form['age'])
        academic_pressure = int(request.form['academic_pressure'])
        study_satisfaction = int(request.form['study_satisfaction'])
        sleep_duration = request.form['sleep_duration']
        dietary_habits = request.form['dietary_habits']
        suicidal_thoughts = request.form['suicidal_thoughts']
        study_hours = int(request.form['study_hours'])
        financial_stress = int(request.form['financial_stress'])
        family_history = request.form['family_history']

        # Preprocess input data
        input_data = preprocess_input(gender, age, academic_pressure, study_satisfaction, sleep_duration, dietary_habits, suicidal_thoughts, study_hours, financial_stress, family_history)

        # Make prediction
        prediction = model.predict(input_data)[0]

        if target_encoder:
          prediction = target_encoder.inverse_transform([int(prediction)])[0] 

        return render_template('results.html', prediction=prediction)

def preprocess_input(gender, age, academic_pressure, study_satisfaction, sleep_duration, dietary_habits, suicidal_thoughts, study_hours, financial_stress, family_history):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Academic Pressure': [academic_pressure],
        'Study Satisfaction': [study_satisfaction],
        'Sleep Duration': [sleep_duration],
        'Dietary Habits': [dietary_habits],
        'Have you ever had suicidal thoughts ?': [suicidal_thoughts],
        'Study Hours': [study_hours],
        'Financial Stress': [financial_stress],
        'Family History of Mental Illness': [family_history]
    })

    for col, encoder in feature_encoders.items():
        if col in input_df:
            input_df[col] = encoder.transform(input_df[col])

    # Ensure numerical columns match the scaler training
    numerical_cols = ['Age', 'Academic Pressure', 'Study Satisfaction', 
                      'Study Hours', 'Financial Stress']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Convert to NumPy array for prediction
    input_data_transformed = input_df.values  # No need to reshape here
    return input_data_transformed

if __name__ == '__main__':
    app.run(debug=True)