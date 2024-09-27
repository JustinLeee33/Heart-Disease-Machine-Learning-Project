# In main.py
from models.logistic_regression import lr_train_and_evaluate
from models.decision_tree import dt_train_and_evaluate
from models.random_forest import rf_train_and_evaluate
from models.gradient_boosting import gb_train_and_evaluate
from models.svm import svm_train_and_evaluate
from visualizations.plots import plot_precision_recall_curves

def main():
    # Load your dataset and preprocess it
    # Assume X_train, X_test, y_train, y_test are defined here
    
    # Define a dictionary of models
    models = {
        'Logistic Regression': lr_train_and_evaluate,
        'Decision Tree': dt_train_and_evaluate,
        'Random Forest': rf_train_and_evaluate,
        'Gradient Boosting': gb_train_and_evaluate,
        'SVM': svm_train_and_evaluate
    }

    # Plot Precision-Recall curves for all models
    plot_precision_recall_curves(models, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
