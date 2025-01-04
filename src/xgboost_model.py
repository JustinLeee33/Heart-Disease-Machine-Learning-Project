from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import os

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning."""
    
    # Log the start of the process
    print(colored("\n--- XGBoost Training and Evaluation with Hyperparameter Tuning Started ---", "green"))
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Step 1: Handle Missing Data (if needed)
    if X_train.isnull().sum().any() or X_test.isnull().sum().any():
        print(colored("Missing values found! Imputing with the median value.", "yellow"))
        X_train.fillna(X_train.median(), inplace=True)
        X_test.fillna(X_test.median(), inplace=True)

    # Step 2: Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Define Hyperparameter Grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_lambda': [0, 1, 10],
        'reg_alpha': [0, 0.5, 1]
    }

    # Step 4: Perform GridSearchCV to find the best hyperparameters
    print(colored("Performing GridSearchCV...", "cyan"))
    grid_search = GridSearchCV(xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss'), param_grid, cv=5, verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    # Step 5: Print Best Parameters and Best Score
    print(colored(f"Best Parameters: {grid_search.best_params_}", "green"))
    print(colored(f"Best Score from GridSearchCV: {grid_search.best_score_:.4f}", "green"))

    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Step 6: Train the Best Model
    print(colored("Training the best model...", "cyan"))
    best_model.fit(X_train_scaled, y_train)

    # Step 7: Evaluate the Best Model
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # F1 Score for better balance between precision and recall
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled), multi_class='ovr')

    print(colored(f"Optimized XGBoost Accuracy: {accuracy:.4f}", "magenta"))
    print(colored(f"F1 Score: {f1:.4f}", "magenta"))
    print(colored(f"ROC-AUC Score: {roc_auc:.4f}", "magenta"))

    # Step 8: Save the Classification Report
    report = classification_report(y_test, y_pred)
    report_path = os.path.join(plot_dir, 'xgboost_best_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(colored(f"Classification report saved to {report_path}.", "green"))

    # Step 9: Cross-Validation to Check Model Stability
    cross_val_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(colored(f"Cross-validation scores: {cross_val_scores}", "cyan"))
    print(colored(f"Mean cross-validation accuracy: {cross_val_scores.mean():.4f}", "cyan"))
    
    # Step 10: Log the End of the Process
    print(colored("\n--- XGBoost Training and Evaluation Completed ---", "green"))
    
    # Return the best model, accuracy, and F1 score
    return best_model, accuracy, f1, roc_auc
