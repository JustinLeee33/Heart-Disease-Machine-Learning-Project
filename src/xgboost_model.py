import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import os
from termcolor import colored  # Import for color logging

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning and early stopping."""
    
    # Log the start of the process
    print(colored("\n--- XGBoost Training and Evaluation Started ---", "green"))
    
    # Define hyperparameters to explore
    print(colored("Defining hyperparameter grid for tuning...", "cyan"))
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [0.5, 1, 1.5]
    }

    # Initialize the model
    print(colored("Initializing the XGBoost model...", "cyan"))
    model = xgb.XGBClassifier(
        eval_metric='mlogloss',
        objective='multi:softprob'
    )

    # Hyperparameter tuning with Randomized Search
    print(colored("Starting Randomized Search for hyperparameter tuning...", "cyan"))
    randomized_search = RandomizedSearchCV(
        model, param_distributions=param_grid, scoring='accuracy', cv=3, n_jobs=-1, n_iter=100, verbose=1, random_state=42
    )
    randomized_search.fit(X_train, y_train)
    print(colored("Randomized Search completed. Best hyperparameters found.", "green"))

    # Train the best model with early stopping
    print(colored("Training the best model with early stopping...", "cyan"))
    best_model = randomized_search.best_estimator_
    
    # Train with early stopping
    best_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)
    print(colored("Model training completed.", "green"))

    # Predict and evaluate
    print(colored("Predicting and evaluating model performance...", "cyan"))
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)
    
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

# Example usage (assuming you have your train/test data ready)
# X_train, X_test, y_train, y_test = your_data_split_function()
# model, predictions, scores = xgb_train_and_evaluate(X_train, X_test, y_train, y_test)
