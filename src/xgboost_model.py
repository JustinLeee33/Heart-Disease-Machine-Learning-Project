import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning."""
    
    # Log the start of the process
    print(colored("\n--- XGBoost Training and Evaluation Started ---", "green"))
    
    # Define a smaller hyperparameter grid for tuning (for Randomized Search)
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
        objective='multi:softprob',  # Make sure the objective is defined
    )

    # Randomized Search with Cross Validation
    random_search = RandomizedSearchCV(
        estimator=model,
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
    print(colored(f"Best Hyperparameters from Random Search: {best_params}", "magenta"))

    # Train the best model
    print(colored("Training the best model from Random Search...", "cyan"))
    best_model = random_search.best_estimator_

    # Create DMatrix objects for training and testing
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set evaluation metric and eval_set during training
    evals = [(dtest, 'eval'), (dtrain, 'train')]
    best_model = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=10,  # Optional: stop early if no improvement
    )

    print(colored("Model training completed.", "green"))

    # Predict and evaluate
    print(colored("Predicting and evaluating model performance...", "cyan"))
    y_pred = best_model.predict(dtest)  # Get class labels directly (not probabilities)
    
    # If using multi-class, ensure that we get the highest probability class (argmax)
    if len(np.unique(y_test)) > 2:  # For multi-class problems
        y_pred = np.argmax(y_pred, axis=1)
    
    y_scores = best_model.predict(dtest)  # Keep probabilities for the evaluation

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
