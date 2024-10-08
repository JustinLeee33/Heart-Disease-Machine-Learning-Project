from src.preprocessing import process_data, convert_categorical_to_int
from src.download import download_and_extract_dataset
from src.visualization import visualize_data
from src.logistic_regression import lr_train_and_evaluate
from src.decision_tree import dt_train_and_evaluate
from src.random_forest import rf_train_and_evaluate
from src.gradient_boosting import gb_train_and_evaluate
from src.svm import svm_train_and_evaluate
from src.XG_Boost import xgb_train_and_evaluate

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def plot_precision_recall_curves(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Plots Precision vs Recall for all models and computes precision and recall from confusion matrix."""
    models = {
        'Logistic Regression': lr_train_and_evaluate,
        'Decision Tree': dt_train_and_evaluate,
        'Random Forest': rf_train_and_evaluate,
        'Gradient Boosting': gb_train_and_evaluate,
        'SVM': svm_train_and_evaluate
    }
    
    plt.figure(figsize=(10, 6))
    
    for model_name, model_func in models.items():
        # Train the model and get predictions
        model, y_pred = model_func(X_train, X_test, y_train, y_test, plot_dir)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()  # Unpack confusion matrix
        
        # Calculate precision and recall from the confusion matrix
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        
        # Print confusion matrix, precision, and recall
        print(f"{model_name} Confusion Matrix:\n{cm}")
        print(f"{model_name} Precision: {precision:.2f}")
        print(f"{model_name} Recall: {recall:.2f}")
        
        # Calculate precision and recall curve
        y_scores = model.predict_proba(X_test)[:, 1]  # Probability estimates for positive class
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
        
        # Plot Precision vs Recall curve
        plt.plot(recall_curve, precision_curve, label=model_name)
    
    plt.title('Precision vs Recall for Different Models')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'precision_recall_curves.png'))
    plt.close()

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
    
    # Plot Precision vs Recall curves for all models
    plot_precision_recall_curves(X_train, X_test, y_train, y_test)

# Call the main function
if __name__ == '__main__':
    main()
