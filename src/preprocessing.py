import pandas as pd
import numpy as np

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

def process_data(file_path):
    """Load and clean the data, then return the cleaned DataFrame."""
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
    
    return data

