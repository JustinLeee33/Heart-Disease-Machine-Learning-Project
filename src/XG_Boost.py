import xgboost as xgb

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model."""
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Save results
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Save classification report and accuracy plot as before
    return model  # Return the trained model
