from src.preprocessing import process_data, convert_categorical_to_int
from src.download import download_and_extract_dataset
from src.visualization import visualize_data
from src.logistic_regression import lr_train_and_evaluate
from src.decision_tree import dt_train_and_evaluate
from src.random_forest import rf_train_and_evaluate
from src.gradient_boosting import gb_train_and_evaluate
from src.svm import svm_train_and_evaluate

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt
import os

def plot_precision_recall_for_model(model, X_test, y_test, model_name):
    """
    Plot precision-recall curve for a single model
    """
    plt.figure(figsize=(8, 6))
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]  # For binary classification
    else:
        # Use decision function for models like SVM
        y_scores = model.decision_function(X_test)
    
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig(f'data/plots/precision_recall_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

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
    
    # Train and evaluate models, and return trained models
    lr_model = lr_train_and_evaluate(X_train, X_test, y_train, y_test)
    dt_model = dt_train_and_evaluate(X_train, X_test, y_train, y_test)
    rf_model = rf_train_and_evaluate(X_train, X_test, y_train, y_test)
    gb_model = gb_train_and_evaluate(X_train, X_test, y_train, y_test)
    svm_model = svm_train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Plot Precision-Recall for each model
    plot_precision_recall_for_model(lr_model, X_test, y_test, 'Logistic Regression')
    plot_precision_recall_for_model(dt_model, X_test, y_test, 'Decision Tree')
    plot_precision_recall_for_model(rf_model, X_test, y_test, 'Random Forest')
    plot_precision_recall_for_model(gb_model, X_test, y_test, 'Gradient Boosting')
    plot_precision_recall_for_model(svm_model, X_test, y_test, 'SVM')

if __name__ == "__main__":
    main()
