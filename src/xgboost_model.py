import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import os

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning and early stopping."""

    # Define hyperparameters to explore
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

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        objective='multi:softprob'
    )

    # Hyperparameter tuning with Grid Search
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Train with early stopping
    best_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print results
    print(f"Optimized XGBoost Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    return best_model, y_pred, y_scores
