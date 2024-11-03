import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import os

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning."""

    # Define hyperparameters for tuning
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',         # Multiclass log-loss
        objective='multi:softprob',     # Multiclass probability output
        max_depth=6,                    # Maximum tree depth
        learning_rate=0.1,              # Step size shrinkage
        n_estimators=100,               # Number of boosting rounds
        subsample=0.8,                  # Fraction of samples per tree
        colsample_bytree=0.8,           # Fraction of features per tree
        gamma=1,                        # Minimum loss reduction for split
        reg_alpha=0,                    # L1 regularization
        reg_lambda=1                    # L2 regularization
    )
    
    # Train the model
    model.fit(X_train, y_train)

    # Predict class labels
    y_pred = model.predict(X_test)

    # Get predicted probabilities for all classes
    y_scores = model.predict_proba(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print results
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    return model, y_pred, y_scores
