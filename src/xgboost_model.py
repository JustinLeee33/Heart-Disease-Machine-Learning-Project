from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning."""
    
    # Log the start of the process
    print(colored("\n--- XGBoost Training and Evaluation Started ---", "green"))
    
    # Initialize the model with the best parameters from RandomizedSearchCV
    print(colored("Initializing the XGBoost model...", "cyan"))
    best_model = xgb.XGBClassifier(
        subsample=0.8,
        reg_lambda=1,
        reg_alpha=0.1,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        gamma=0.1,
        colsample_bytree=0.8,
        objective='multi:softmax',  # For multi-class classification
        eval_metric='mlogloss'
    )
    
    # Train the best model
    print(colored("Training the best model...", "cyan"))
    best_model.fit(X_train, y_train)
    print(colored("Model training completed.", "green"))
    
    # Predict and evaluate
    print(colored("Predicting and evaluating model performance...", "cyan"))
    y_pred = best_model.predict(X_test)  # Get predicted class labels
    
    # No need for np.argmax if it's already a class label
    y_scores = best_model.predict_proba(X_test)  # Get predicted probabilities for evaluation
    
    print(colored("Prediction completed.", "green"))
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(colored(f"Optimized XGBoost Accuracy: {accuracy:.4f}", "magenta"))
    print(colored("Classification Report:", "magenta"))
    print(report)
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save the classification report
    report_path = os.path.join(plot_dir, 'xgboost_classification_report.txt')
    print(colored("Saving the classification report...", "cyan"))
    with open(report_path, 'w') as f:
        f.write(report)
    print(colored(f"Classification report saved to {report_path}.", "green"))
    
    # Log the end of the process
    print(colored("\n--- XGBoost Training and Evaluation Completed ---", "green"))
    
    # Return the model, predictions, and predicted probabilities (y_scores)
    return best_model, y_pred, y_scores
