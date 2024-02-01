import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import math

def preprocess_data(data):
    # Drop columns with the same value in every row
    data = data.loc[:, (data != data.iloc[0]).any()]

    # Drop the 'Timestamp' column
    data = data.drop(columns=['Timestamp'], axis=1, errors='ignore')

    # Identify columns with string values
    non_numeric_columns = data.select_dtypes(include=['object']).columns

    # Remove columns with string values before replacing NaN
    data = data.drop(columns=non_numeric_columns)

    # Replace NaN values with the mean of each row
    data = data.apply(lambda x: x.fillna(x.mean()), axis=1)

    # Remove rows with infinite values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Convert float values to ceil using map instead of applymap
    data = data.apply(lambda x: x.map(lambda x: math.ceil(x) if isinstance(x, (float, np.float64)) else x))

    return data

def remove_non_common_columns(train_data, test_data):
    # Find common columns between training and testing datasets
    common_columns = train_data.columns.intersection(test_data.columns)

    # Keep only common columns in both datasets
    train_data = train_data[common_columns]
    test_data = test_data[common_columns]

    return train_data, test_data

def train_and_save_predictions(train_file, test_file, output_file):
    # Load training data with explicit dtype to handle mixed types
    train_data = pd.read_csv(train_file, low_memory=False)

    # Load testing data
    test_data = pd.read_csv(test_file)

    # Remove features not present in both training and testing datasets
    train_data, test_data = remove_non_common_columns(train_data, test_data)

    # Print the list of common columns
    print("Common Columns:")
    print(train_data.columns)

    # Print the columns in the training data
    print("Columns in Training Data:")
    print(train_data.columns)

    # Print the columns in the testing data
    print("Columns in Testing Data:")
    print(test_data.columns)

    # Preprocess training data
    train_data = preprocess_data(train_data)

    # Train the Isolation Forest model
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(train_data)

    # Preprocess testing data
    test_data = preprocess_data(test_data)

    # Ensure that columns in testing data match those used during training
    test_data = test_data[train_data.columns.intersection(test_data.columns)]

    # Predict anomalies in the testing data
    predictions = model.predict(test_data)

    # Create a DataFrame with the predicted port numbers
    result_df = pd.DataFrame({'PortNumber': test_data.index, 'Prediction': predictions})

    # Save the DataFrame to a CSV file on the desktop
    result_df[result_df['Prediction'] == -1].to_csv(output_file, index=False)

if __name__ == "_main_":
    # Replace 'train_data.csv' and 'test_data.csv' with the actual file paths
    train_file_path = r"C:\Users\ravin\OneDrive\Desktop\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
    test_file_path = r"C:\Users\ravin\OneDrive\Desktop\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"

    # Set the output file path for saving predicted port numbers with a value of -1
    output_file_path = r"C:\Users\ravin\OneDrive\Desktop\predictedFiles1.csv"

    train_and_save_predictions(train_file_path, test_file_path, output_file_path)