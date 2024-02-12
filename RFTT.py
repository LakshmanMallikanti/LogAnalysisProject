import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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
        return None, None, None, None

    # Get common columns
    common_columns = list(set(train_data.columns).intersection(set(test_data.columns)))
    
    if len(common_columns) == 0:
        print("No common columns found between training data and test data.")
        return None, None, None, None
    print(common_columns)

    # Extract features and target variable
    X_train = train_data[common_columns].drop(columns=['Label'])  # Exclude the target variable
    y_train = train_data['Label']

    X_test = test_data[common_columns].drop(columns=['Label'])  # Exclude the target variable
    y_test = test_data['Label']

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Predict anomaly
    anomaly_prediction = "Attack happened" if any(y_pred) else "Attack not happened"

    return accuracy, anomaly_prediction, confusion_mat


# Example usage
if __name__ == "__main__":
    train_file = r"C:\Users\laksh\Desktop\CICIDS2019.csv"
    test_file = r"C:\Users\laksh\Desktop\CICIDS2017.csv"
    accuracy, anomaly_prediction, confusion_mat = train_and_evaluate_model(train_file, test_file)
    print("Accuracy:", accuracy)
    print("Anomaly Prediction:", anomaly_prediction)
    print("Confusion Matrix:")
    print(confusion_mat)
