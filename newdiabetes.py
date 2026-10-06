import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_diabetes_data(n_samples=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate demographic information
    ages = np.random.normal(55, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 90)  # Limit age range to realistic values
    
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
    hba1c = np.clip(hba1c, 5.0, 14.0)
    
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
    
    # Determine if the patient has diabetes
    has_diabetes = []
    diabetes_type = []
    for i in range(n_samples):
        if hba1c[i] > 6.5 or fasting_glucose[i] > 126:
            has_diabetes.append(True)
            # Classify diabetes type
            if hba1c[i] > 8.0 or fasting_glucose[i] > 180:
                diabetes_type.append('Type 1')
            elif genders[i] == 'Female' and ages[i] >= 18 and ages[i] <= 45 and fasting_glucose[i] > 140:
                diabetes_type.append('Gestational Diabetes')
            else:
                diabetes_type.append('Type 2')
        else:
            has_diabetes.append(False)
            diabetes_type.append('None')
    
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
        'Has_Diabetes': has_diabetes,
        'Diabetes_Type': diabetes_type
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

# Create pie chart for diabetes type distribution
def create_diabetes_pie_chart(data):
    # Count the occurrences of each diabetes type
    diabetes_counts = data['Diabetes_Type'].value_counts()
    
    # Set up the figure and axis
    plt.figure(figsize=(10, 7))
    
    # Create pie chart
    plt.pie(
        diabetes_counts, 
        labels=diabetes_counts.index,
        autopct='%1.1f%%',  # Show percentages
        startangle=90,      # Start at the top
        shadow=True,        # Add shadow effect
        explode=[0.05] * len(diabetes_counts),  # Slightly explode all slices
        colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']  # Custom colors
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    
    # Add title
    plt.title('Distribution of Diabetes Types', fontsize=16, pad=20)
    
    # Add legend with counts
    legend_labels = [f"{label} ({count})" for label, count in zip(diabetes_counts.index, diabetes_counts)]
    plt.legend(legend_labels, loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Add additional information
    plt.figtext(0.5, 0.01, f"Total Patients: {len(data)}", ha="center", fontsize=12)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('diabetes_type_distribution.png')
    plt.show()
    
    return diabetes_counts

# Call the function to create the pie chart
diabetes_counts = create_diabetes_pie_chart(diabetes_data)
print("\nDiabetes Type Distribution:")
print(diabetes_counts)