from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Column transformer and scaler should be defined as used during training
categorical = ['sex', 'cp', 'restecg', 'slope', 'thal']
do_not_touch = ['fbs', 'exang']
non_categorical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Ensure consistent categories
dataset = pd.read_csv('heart.csv')
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical)],
    remainder='passthrough'
)
X = ct.fit_transform(dataset[categorical + do_not_touch + non_categorical])
scaler = StandardScaler()
scaler.fit(X[:, -6:])

# Home route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    input_data = [data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'], data['fbs'], data['restecg'], data['thalach'], data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']]
    
    # Convert to appropriate types
    input_data = [float(i) for i in input_data]
    
    # Preprocess the input data
    input_df = pd.DataFrame([input_data], columns=non_categorical + do_not_touch + categorical)
    input_transformed = ct.transform(input_df)
    input_transformed[:, -6:] = scaler.transform(input_transformed[:, -6:])
    
    # Predict using the loaded model
    prediction = model.predict(input_transformed)
    
    # Interpret the result
    result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
    
    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
