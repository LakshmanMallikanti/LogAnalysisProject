import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Function to preprocess data
def preprocess_data(df):
    # Replace NaN values with row-wise mean for numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=1)
    
    # Remove rows with infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df


# Function to train and evaluate model
def train_and_evaluate_model(train_file, test_file):
    # Load training data
    train_data = pd.read_csv(train_file)
    print("Number of samples in training data before preprocessing:", len(train_data))
    train_data = preprocess_data(train_data)

    # Load test data
    test_data = pd.read_csv(test_file)
    print("Number of samples in test data before preprocessing:", len(test_data))
    test_data = preprocess_data(test_data)

    print("Number of samples in training data after preprocessing:", len(train_data))
    print("Number of samples in test data after preprocessing:", len(test_data))

    # Check if any samples are left for testing
    if len(test_data) == 0:
        print("No samples remaining in test data after preprocessing. Adjust preprocessing steps.")
        return None

    # Get common numeric columns
    common_numeric_columns = set(train_data.select_dtypes(include=[np.number]).columns).intersection(
        set(test_data.select_dtypes(include=[np.number]).columns)
    )
    
    common_numeric_columns=list(common_numeric_columns)

    if len(common_numeric_columns) == 0:
        print("No common numeric columns found between training and test data.")
        return None
    
    # Extract features and target variable
    X_train = train_data[common_numeric_columns]
    y_train = train_data['Label']  # Assuming the target variable is 'Label'

    X_test = test_data[common_numeric_columns]

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Predict anomaly
    anomaly_prediction = "Attack happened" if any(y_pred) else "Attack not happened"
    return anomaly_prediction


# Example usage
if __name__ == "__main__":
    train_file = r"C:\Users\laksh\Desktop\CICIDS20181.csv"
    test_file = r"C:\Users\laksh\Desktop\CICIDS20192.csv"
    anomaly_prediction = train_and_evaluate_model(train_file, test_file)
    print("Anomaly Prediction:", anomaly_prediction)