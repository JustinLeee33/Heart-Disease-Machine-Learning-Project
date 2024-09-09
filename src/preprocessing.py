import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the dataset by removing duplicates and handling missing values."""
    data.drop_duplicates(inplace=True)
    missing_values = data.isnull().sum()
    
    # Identify columns with numeric data
    numeric_columns = data.select_dtypes(include=np.number).columns
    
    # Fill missing values only in numeric columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    
    return data, missing_values

def transform_data(data, target_column):
    """Transform the data by scaling numerical features and encoding categorical features."""
    # Separate features and target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def process_data(file_path, target_column):
    """Load, clean, transform, and split the data, and return the processed DataFrame."""
    data = load_data(file_path)
    data, missing_values = clean_data(data)
    
    # Print missing values and other information
    print("Null values per column:")
    print(missing_values)
    
    print("\nColumns with null values:")
    print(missing_values[missing_values > 0])
    
    categorical_features = data.select_dtypes(include=['object']).columns
    print(f"Number of categorical features: {len(categorical_features)}")
    print("Categorical features:")
    print(categorical_features)
    
file_path = 'heart_disease_uci.csv'
target_column = 'num'  # Replace with the actual name of the target variable column
X_train, X_test, y_train, y_test = process_data(file_path, target_column)
