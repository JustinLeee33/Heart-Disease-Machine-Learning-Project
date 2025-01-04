from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import itertools
import os

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning."""
    
    # Log the start of the process
    print(colored("\n--- XGBoost Training and Evaluation with Hyperparameter Tuning Started ---", "green"))
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Define hyperparameter ranges
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_lambda': [0, 1, 10],
        'reg_alpha': [0, 0.5, 1]
    }
    
    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['learning_rate'],
        param_grid['subsample'],
        param_grid['colsample_bytree'],
        param_grid['gamma'],
        param_grid['reg_lambda'],
        param_grid['reg_alpha']
    ))
    
    print(colored(f"Total parameter combinations to try: {len(param_combinations)}", "cyan"))
    
    # Initialize tracking variables
    best_accuracy = 0
    best_params = None
    best_model = None
    
    # Iterate over all combinations
    for idx, params in enumerate(param_combinations):
        print(colored(f"Training with parameter set {idx + 1}/{len(param_combinations)}...", "cyan"))
        
        # Extract parameters
        n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_lambda, reg_alpha = params
        
        # Initialize the model with current parameters
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective='multi:softmax',
            eval_metric='mlogloss'
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(colored(f"Accuracy for this set: {accuracy:.4f}", "yellow"))
        
        # Track the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model
            
    # Print the best parameters and accuracy
    print(colored("\nBest Parameters:", "green"))
    print(best_params)
    print(colored(f"Best Accuracy: {best_accuracy:.4f}", "magenta"))
    
    # Save the best model's classification report
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    report_path = os.path.join(plot_dir, 'xgboost_best_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(colored(f"Best model's classification report saved to {report_path}.", "green"))
    
    # Return the best model and performance
    return best_model, best_accuracy, best_params
