from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import os

def xgb_tune_and_evaluate(X, y, plot_dir='data/plots', target_accuracy=0.9):
    """Tune and evaluate XGBoost model for optimal accuracy."""
    
    # Log the start of the process
    print(colored("\n--- XGBoost Hyperparameter Tuning and Evaluation Started ---", "green"))
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_lambda': [0, 1, 10],
        'reg_alpha': [0, 0.5, 1],
    }
    
    # Initialize the base model
    base_model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', random_state=42)
    
    # Use RandomizedSearchCV for hyperparameter tuning
    print(colored("Starting hyperparameter tuning with RandomizedSearchCV...", "cyan"))
    search = RandomizedSearchCV(
        base_model, 
        param_distributions=param_grid, 
        n_iter=50, 
        scoring='accuracy', 
        cv=3, 
        verbose=2, 
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    # Extract the best model and parameters
    best_model = search.best_estimator_
    best_params = search.best_params_
    print(colored(f"Best Hyperparameters: {best_params}", "magenta"))
    
    # Evaluate the best model
    print(colored("Evaluating the best model on the test set...", "cyan"))
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(colored(f"Final Model Accuracy: {accuracy:.4f}", "magenta"))
    print(colored("Classification Report:", "magenta"))
    print(report)
    
    # Save the classification report
    report_path = os.path.join(plot_dir, 'xgboost_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(colored(f"Classification report saved to {report_path}.", "green"))
    
    # Save the best parameters
    params_path = os.path.join(plot_dir, 'best_hyperparameters.txt')
    with open(params_path, 'w') as f:
        f.write(str(best_params))
    print(colored(f"Best hyperparameters saved to {params_path}.", "green"))
    
    # Log the end of the process
    print(colored("\n--- XGBoost Hyperparameter Tuning and Evaluation Completed ---", "green"))
    
    # Return the best model, predictions, and probabilities
    return best_model, y_pred, y_scores
