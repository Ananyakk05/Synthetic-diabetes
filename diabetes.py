import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def generate_diabetes_data(n_samples=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate demographic information
    ages = np.random.normal(55, 15, n_samples).astype(int) 
    ages = np.clip(ages, 4, 90)  # Limit age range to realistic values
    
    # Gender distribution
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
    
    # BMI - normal distribution with realistic ranges for diabetes patients
    bmi = np.random.normal(29.5, 6, n_samples)
    bmi = np.clip(bmi, 18.5, 45)
    
    # Blood Pressure (systolic/diastolic)
    systolic_bp = np.random.normal(135, 15, n_samples).astype(int)
    systolic_bp = np.clip(systolic_bp, 90, 180)
    diastolic_bp = np.random.normal(85, 10, n_samples).astype(int)
    diastolic_bp = np.clip(diastolic_bp, 60, 100)
    
    # HbA1c levels (%)
    hba1c = np.random.normal(7.5, 1.5, n_samples)
    hba1c = np.clip(hba1c, 3.0, 14.0)
    
    # Fasting Blood Glucose (mg/dL)
    fasting_glucose = np.random.normal(140, 30, n_samples)
    fasting_glucose = np.clip(fasting_glucose, 70, 250)
    
    # Random Blood Glucose (mg/dL)
    random_glucose = np.random.normal(170, 50, n_samples)
    random_glucose = np.clip(random_glucose, 70, 300)
    
    # Cholesterol levels
    total_cholesterol = np.random.normal(200, 40, n_samples)
    hdl = np.random.normal(45, 10, n_samples)
    ldl = np.random.normal(120, 30, n_samples)
    triglycerides = np.random.normal(150, 50, n_samples)
    
    # Family history
    family_history = np.random.choice([True, False], n_samples, p=[0.4, 0.6])
    
    # Diabetes type
    diabetes_type = np.random.choice(['Type 1', 'Type 2', 'Gestational'], n_samples, p=[0.1, 0.85, 0.05])
    
    # Treatment
    treatments = []
    for d_type in diabetes_type:
        if d_type == 'Type 1':
            treatments.append('Insulin')
        elif d_type == 'Type 2':
            treatments.append(np.random.choice(['Metformin', 'Insulin', 'Sulfonylureas', 'Combined Therapy'], p=[0.4, 0.2, 0.2, 0.2]))
        else:
            treatments.append(np.random.choice(['Diet and Exercise', 'Insulin'], p=[0.6, 0.4]))
    
    # Create DataFrame
    data = pd.DataFrame({
        'PatientID': range(1, n_samples + 1),
        'Age': ages,
        'Gender': genders,
        'BMI': np.round(bmi, 1),
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp,
        'HbA1c': np.round(hba1c, 1),
        'Fasting_Glucose': np.round(fasting_glucose, 1),
        'Random_Glucose': np.round(random_glucose, 1),
        'Total_Cholesterol': np.round(total_cholesterol, 1),
        'HDL': np.round(hdl, 1),
        'LDL': np.round(ldl, 1),
        'Triglycerides': np.round(triglycerides, 1),
        'Family_History': family_history,
        'Diabetes_Type': diabetes_type,
        'Treatment': treatments
    })
    
    return data

# Generate the synthetic data
diabetes_data = generate_diabetes_data(1000)

# Save to CSV
diabetes_data.to_csv('synthetic_diabetes_data.csv', index=False)

# Display first few rows and basic statistics
print("\nFirst few rows of the generated data:")
print(diabetes_data.head())
print("\nBasic statistics of the numerical columns:")
print(diabetes_data.describe())

def preprocess_data(data):
    # Convert gender to numerical (0 for Female, 1 for Male)
    data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
    
    # Separate features and target
    X = data[['Age', 'Gender', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
              'HbA1c', 'Fasting_Glucose', 'Random_Glucose', 
              'Total_Cholesterol', 'HDL', 'LDL', 'Triglycerides']]
    y = data['Diabetes_Type']
    
    return X, y

def train_diabetes_classifier(data):
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    joblib.dump(model, 'diabetes_classifier.joblib')
    joblib.dump(scaler, 'diabetes_scaler.joblib')
    
    return model, scaler

def predict_diabetes_type(patient_data, model, scaler):
    # Convert gender to numerical
    patient_data['Gender'] = 0 if patient_data['Gender'].lower() == 'female' else 1
    
    # Create feature array in the correct order
    features = ['Age', 'Gender', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
                'HbA1c', 'Fasting_Glucose', 'Random_Glucose', 
                'Total_Cholesterol', 'HDL', 'LDL', 'Triglycerides']
    
    X = np.array([[patient_data[feature] for feature in features]])
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    return prediction, probabilities

# Example usage
if __name__ == "__main__":
    # Load your diabetes dataset
    data = pd.read_csv('synthetic_diabetes_data.csv')
    
    # Train the classifier
    model, scaler = train_diabetes_classifier(data)
    
    # Example patient data
    patient = {
        'Age': 45,
        'Gender': 'Female',
        'BMI': 28.5,
        'Systolic_BP': 130,
        'Diastolic_BP': 85,
        'HbA1c': 6.8,
        'Fasting_Glucose': 120,
        'Random_Glucose': 150,
        'Total_Cholesterol': 190,
        'HDL': 45,
        'LDL': 120,
        'Triglycerides': 150
    }
    
    # Make prediction
    prediction, probabilities = predict_diabetes_type(patient, model, scaler)
    
    print("\nPrediction Results:")
    print(f"Predicted Diabetes Type: {prediction}")
    print("\nPrediction Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{model.classes_[i]}: {prob:.2%}")
