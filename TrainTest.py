import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

# Load the dataset
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Preprocess the dataset
def preprocess_data(dataset):
    # Calculate and print the number of NaN rows
    nan_rows = dataset.isnull().sum(axis=1)
    print(f"Number of NaN rows: {nan_rows.sum()}")

    # Calculate and print the number of duplicate rows
    duplicate_rows = dataset.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")

    # Replace NaN values with the mean value of each row
    numeric_columns = dataset.select_dtypes(include=np.number).columns
    dataset[numeric_columns] = dataset[numeric_columns].apply(lambda row: row.fillna(row.mean()), axis=1)


    # Remove identical rows
    dataset = dataset.drop_duplicates()

    # Drop non-relevant columns (e.g., date-time column)
    dataset = dataset.drop(columns=['Timestamp'], axis=1, errors='ignore')

    # Drop columns with all the same values
    dataset = dataset.loc[:, dataset.nunique() != 1]

    # Check for and handle infinite or large values
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(inplace=True)

    # Make sure the data is in the right format for training
    return dataset

# Train the model with hyperparameter tuning and parallelization
def train_model(X_train, y_train):
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    end_time = time.time()
    time_taken = end_time - start_time
    return model, time_taken

# Test the model
def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=1)
    confusion_mat = confusion_matrix(y_test, predictions)
    return accuracy, report, confusion_mat

# Main function
def main():
    # Replace 'your_dataset.csv' with the actual path to your CSV file
    file_path = r"C:\Users\laksh\Desktop\CICIDS2019.csv"
    
    # Load and preprocess the dataset
    dataset = load_dataset(file_path)
    dataset = preprocess_data(dataset)

    # Print the number of rows and columns in the dataset
    print(f"Number of rows: {len(dataset)}, Number of columns: {len(dataset.columns)}")

    # Assign the name of the last column of the dataset to 'label_column'
    label_column = dataset.columns[-1]

    # Label encoding for the label column
    label_encoder = LabelEncoder()
    dataset[label_column] = label_encoder.fit_transform(dataset[label_column])

    # Split the dataset into features (X) and target variable (y)
    X = dataset.drop(label_column, axis=1)
    y = dataset[label_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model and measure time taken
    model, time_taken = train_model(X_train, y_train)

    # Test the model
    accuracy, report, confusion_mat = test_model(model, X_test, y_test)

    # Print the results
    print(f'Accuracy: {accuracy* 100:.5f}')
    print('Classification Report:')
    print(report)
    
    # Print the confusion matrix
    print('Confusion Matrix:')
    print(confusion_mat)

    # Print the time taken to train the model
    print(f"Time taken to train the model: {time_taken} seconds")

if __name__ == "__main__":
    main()













