import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Function to preprocess data
def preprocess_data(df):
    # Replace NaN values with row-wise mean for numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=1)
    
    # Remove rows with infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df
from sklearn.model_selection import train_test_split

# Modify the train_and_evaluate_model function
def train_and_evaluate_model(train_file):
    # Load data
    data = pd.read_csv(train_file)
    
    # Preprocess data
    data = preprocess_data(data)

    # Split data into features and target variable
    X = data.drop(columns=['Label'])
    y = data['Label']

    # Instantiate LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the target variable
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate precision
    precision = precision_score(y_test, y_pred, average=None)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print precision table
    print("Precision Table:")
    print("Class\tPrecision")
    for i, prec in enumerate(precision):
        print(f"{i}\t{prec}")

    # Print accuracy
    print("Accuracy:", accuracy)

    # Predict anomaly
    anomaly_prediction = "Attack happened" if any(y_pred) else "Attack not happened"
    return anomaly_prediction

# Example usage
if __name__ == "__main__":
    train_file = r"C:\Users\laksh\Desktop\Final1.csv"
    anomaly_prediction = train_and_evaluate_model(train_file)
    print("Anomaly Prediction:", anomaly_prediction)
