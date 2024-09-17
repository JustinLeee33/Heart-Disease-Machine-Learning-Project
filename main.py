from src.download import download_and_extract_dataset
from src.preprocessing import process_data
from src.visualization import visualize_data
import os
from src.logistic_regression import train_and_evaluate as lr_train_and_evaluate
from src.decision_tree import train_and_evaluate as dt_train_and_evaluate
from src.random_forest import train_and_evaluate as rf_train_and_evaluate
from src.gradient_boosting import train_and_evaluate as gb_train_and_evaluate
from src.svm import train_and_evaluate as svm_train_and_evaluate

def main():
    print('Here')
    # Download the Heart Disease Dataset from Kaggle (save to data/csv/heart_disease_uci.csv)
    download_and_extract_dataset()

    # Gather our Data (import the csv)
    data = process_data('data/csv/heart_disease_uci.csv')

    # Visualize the Data (save into a folder called data/plots)
    visualize_data(data)

    # Split the data into features and target variable
    from sklearn.model_selection import train_test_split
    
    # Assuming the target column is named 'target'
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create data/plots directory if it doesn't exist
    os.makedirs('data/plots', exist_ok=True)
    
    # Train and evaluate models
    lr_train_and_evaluate(X_train, X_test, y_train, y_test)
    dt_train_and_evaluate(X_train, X_test, y_train, y_test)
    rf_train_and_evaluate(X_train, X_test, y_train, y_test)
    gb_train_and_evaluate(X_train, X_test, y_train, y_test)
    svm_train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
