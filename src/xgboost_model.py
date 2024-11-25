import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import os
from termcolor import colored
import warnings

# Suppress specific XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Parameters: { \"use_label_encoder\" } are not used.*")

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning."""
    
    # Log the start of the process
    print(colored("\n--- XGBoost Training and Evaluation Started ---", "green"))
    
    # Define a smaller hyperparameter grid for tuning
    print(colored("Defining hyperparameter grid for tuning...", "cyan"))
    param_dist = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.1, 0.01, 0.001], 
        'n_estimators': [100, 200],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 1],
        'reg_alpha': [0, 0.5],
        'reg_lambda': [0.5, 1]
    }

    # Initialize the model
    print(colored("Initializing the XGBoost model...", "cyan"))
    model = xgb.XGBClassifier(
        eval_metric='mlogloss',
        objective='multi:softprob'
    )

    # Hyperparameter tuning with Randomized Search
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=50,  # Number of random iterations
        scoring='accuracy',  # Choose your scoring metric
        cv=3,  # Number of cross-validation folds
        n_jobs=-1,  # Use all available cores
        verbose=2,  # Increase verbosity
    )

    # Perform the randomized search
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = random_search.best_params_
    # Train the best model
    print(colored("Training the best model...", "cyan"))
    best_model = randomized_search.best_estimator_
    
    # Include eval_set and eval_metric during training
    best_model.fit(X_train, y_train, eval_metric='mlogloss', eval_set=[(X_test, y_test)], verbose=True)  # verbose=True will print the progress
    print(colored("Model training completed.", "green"))

    xgb_model = XGBClassifier(
        # ...your hyperparameters
        early_stopping_rounds=10,  # Stop if validation score doesn't improve for 10 rounds
        eval_set=[(X_val, y_val)],  # Validation data
        eval_metric='logloss',  # Choose your evaluation metric
    )
    xgb_model.fit(X_train, y_train)

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
