import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Function to preprocess data
def preprocess_data(df):
    # Drop columns with non-numeric data
    df = df.select_dtypes(include=[np.number])

    # Replace NaN values with row-wise mean for numerical columns
    df = df.apply(lambda x: x.fillna(x.mean()), axis=1)
    
    # Remove rows with infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df

# Modify the train_and_evaluate_model function
def train_and_evaluate_model(train_file, test_file):
    try:
        # Load training data
        train_data = pd.read_csv(train_file)
        
        # Load testing data
        test_data = pd.read_csv(test_file)

        # Combine training and testing data for preprocessing
        combined_data = pd.concat([train_data, test_data])

        # Preprocess combined data
        combined_data = preprocess_data(combined_data)

        # Split combined data into features and target variable
        X_train = combined_data.drop(columns=['Label'])
        y_train = combined_data['Label']

        # Instantiate LabelEncoder
        label_encoder = LabelEncoder()

        # Encode the target variable
        y_train_encoded = label_encoder.fit_transform(y_train)

        # Train the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train_encoded)

        # Make predictions on testing data
        test_data_processed = combined_data[len(train_data):]  # Extract testing data
        X_test = test_data_processed.drop(columns=['Label'])
        y_test_pred = model.predict(X_test)

        # Predict anomaly based on testing data
        anomaly_prediction = "Anomaly Detected" if any(y_test_pred == 1) else "No Anomaly Detected"

        return anomaly_prediction

    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    train_file = r"C:\Users\laksh\Desktop\Final1.csv"
    test_file = r"C:\Users\laksh\Desktop\test2.csv"
    anomaly_prediction = train_and_evaluate_model(train_file, test_file)
    print("Anomaly Prediction:", anomaly_prediction)
