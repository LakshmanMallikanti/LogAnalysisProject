import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(df):
    """Preprocesses the dataset by handling missing and infinite values."""
    if df.empty:
        print("Warning: Empty dataset received for preprocessing.")
        return df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=1)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

def train_and_evaluate_model(train_file, test_file):
    """Trains and evaluates a RandomForest model for anomaly detection."""
    # Load training data
    try:
        train_data = pd.read_csv(train_file)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None
    
    print("Training data samples before preprocessing:", len(train_data))
    train_data = preprocess_data(train_data)
    
    # Ensure 'Label' column exists in training data
    if 'Label' not in train_data.columns:
        print("Error: 'Label' column missing in training data.")
        return None
    
    # Load test data
    try:
        test_data = pd.read_csv(test_file)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None
    
    print("Test data samples before preprocessing:", len(test_data))
    test_data = preprocess_data(test_data)
    
    print("Training data samples after preprocessing:", len(train_data))
    print("Test data samples after preprocessing:", len(test_data))
    
    if test_data.empty:
        print("Error: No valid test samples remaining after preprocessing.")
        return None
    
    # Find common numeric columns
    common_numeric_columns = list(set(train_data.select_dtypes(include=[np.number]).columns) &
                                  set(test_data.select_dtypes(include=[np.number]).columns))
    
    if not common_numeric_columns:
        print("Error: No common numeric features found between training and test data.")
        return None
    
    # Extract features and target variable
    X_train = train_data[common_numeric_columns]
    y_train = train_data['Label']
    X_test = test_data[common_numeric_columns]
    
    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Predict anomaly
    anomaly_prediction = "Attack happened" if any(y_pred) else "Attack not happened"
    return anomaly_prediction

if __name__ == "__main__":
    train_file = r"C:\Users\laksh\Desktop\TRAINING.csv"  # Ensure this is a valid CSV file
    test_file = r"C:\Users\laksh\Desktop\TRAINING.csv"    # Ensure this is a valid CSV file
    
    result = train_and_evaluate_model(train_file, test_file)
    if result:
        print("Anomaly Prediction:", result)
