from src.preprocessing import process_data, convert_categorical_to_int
from src.download import download_and_extract_dataset
from src.visualization import visualize_data
from src.logistic_regression import lr_train_and_evaluate
from src.decision_tree import dt_train_and_evaluate
from src.random_forest import rf_train_and_evaluate
from src.gradient_boosting import gb_train_and_evaluate
from src.svm import svm_train_and_evaluate

from sklearn.model_selection import train_test_split

import os

def main():
    # Download the Heart Disease Dataset from Kaggle (save to data/csv/heart_disease_uci.csv)
    download_and_extract_dataset()

    # Gather our Data (import the csv)
    data = process_data('data/csv/heart_disease_uci.csv')

    # Visualize the Data (save into a folder called data/plots)
    visualize_data(data)
    
    # Convert categorical data to integers
    converted_data, category_mappings = convert_categorical_to_int(data)

    # Print category mappings
    print("Category Mappings:\n", category_mappings)
    
    # Drop the 'num' column (ground truth) for feature data (X)
    if 'num' in converted_data.columns:
        X = converted_data.drop(columns=['num'])
        y = converted_data['num']
    else:
        raise KeyError("The 'num' column (ground truth) is missing from the dataset.")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ensure the data/plots directory exists
    os.makedirs('data/plots', exist_ok=True)
    
    # Train and evaluate models
    lr_train_and_evaluate(X_train, X_test, y_train, y_test)
    dt_train_and_evaluate(X_train, X_test, y_train, y_test)
    rf_train_and_evaluate(X_train, X_test, y_train, y_test)
    gb_train_and_evaluate(X_train, X_test, y_train, y_test)
    svm_train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
