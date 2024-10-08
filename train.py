# main.py

from src.preprocessing import process_data, convert_categorical_to_int
from src.download import download_and_extract_dataset
from src.visualization import visualize_data
from src.logistic_regression import lr_train_and_evaluate
from src.decision_tree import dt_train_and_evaluate
from src.random_forest import rf_train_and_evaluate
from src.gradient_boosting import gb_train_and_evaluate
from src.svm import svm_train_and_evaluate
from src.xgboost_model import xgb_train_and_evaluate  # Renamed module for clarity
from src.ensemble_learning import ensemble_train_and_evaluate

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_precision_recall_curves(y_test, models_scores, plot_dir='data/plots'):
    """Plots Precision vs Recall for all models."""
    plt.figure(figsize=(10, 6))

    for model_name, y_scores in models_scores.items():
        # Calculate precision and recall curve
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
    # Ensure the data/plots directory exists
    os.makedirs('data/plots', exist_ok=True)

    # Download the Heart Disease Dataset from Kaggle
    download_and_extract_dataset()

    # Import and process the data
    data = process_data('data/csv/heart_disease_uci.csv')
    visualize_data(data)
    converted_data, category_mappings = convert_categorical_to_int(data)
    print("Category Mappings:\n", category_mappings)
    
    # Separate features and target
    if 'num' in converted_data.columns:
        X = converted_data.drop(columns=['num'])
        y = converted_data['num']
    else:
        raise KeyError("The 'num' column (ground truth) is missing from the dataset.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Logistic Regression': lr_train_and_evaluate,
        'Decision Tree': dt_train_and_evaluate,
        'Random Forest': rf_train_and_evaluate,
        'Gradient Boosting': gb_train_and_evaluate,
        'SVM': svm_train_and_evaluate,
        'XGBoost': xgb_train_and_evaluate
        'Ensemble Learning': ensemble_train_and_evaluate,
    }

    # Store scores for plotting
    models_scores = {}

    # Train and evaluate models
    for model_name, model_func in models.items():
        print(f"Training and evaluating {model_name}...")
        model, y_pred, y_scores = model_func(X_train, X_test, y_train, y_test, plot_dir='data/plots')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        # Print results
        print(f"{model_name} Confusion Matrix:\n{cm}")
        print(f"{model_name} Precision: {precision:.2f}")
        print(f"{model_name} Recall: {recall:.2f}\n")

        # Store scores
        models_scores[model_name] = y_scores

    # Plot Precision vs Recall curves
    plot_precision_recall_curves(y_test, models_scores, plot_dir='data/plots')

if __name__ == '__main__':
    main()
