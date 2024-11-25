def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots', random_state=42):
    """Train and evaluate XGBoost model with hyperparameter tuning and early stopping."""

    print(colored("\n--- XGBoost Training and Evaluation Started ---", "green"))

    # Define hyperparameter grid for tuning
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

    # Initialize the XGBoost model
    print(colored("Initializing the XGBoost model...", "cyan"))
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        use_label_encoder=False,  # Suppress deprecated warning
        random_state=random_state
    )

    # Perform hyperparameter tuning with Randomized Search
    print(colored("Starting hyperparameter tuning with RandomizedSearchCV...", "cyan"))
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,  # Number of random iterations
        scoring='accuracy',  # Choose your scoring metric
        cv=3,  # Number of cross-validation folds
        n_jobs=-1,  # Use all available cores
        verbose=2,  # Increase verbosity for debugging
        random_state=random_state  # Ensures reproducibility
    )

    # Perform randomized search
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters and train the best model
    best_params = random_search.best_params_
    print(colored(f"Best Hyperparameters: {best_params}", "magenta"))
    best_model = random_search.best_estimator_

    # Train the best model with early stopping
    print(colored("Training the best model...", "cyan"))
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],  # Validation set for early stopping
        early_stopping_rounds=10,  # Stop if no improvement in 10 rounds
        verbose=True
    )
    print(colored("Model training completed.", "green"))

    # Predict and evaluate model performance
    print(colored("Predicting and evaluating model performance...", "cyan"))
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)

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

    print(colored("\n--- XGBoost Training and Evaluation Completed ---", "green"))

    return best_model, y_pred, y_scores
