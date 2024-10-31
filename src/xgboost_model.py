import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate an optimized XGBoost model with hyperparameter tuning and feature importance plot."""
    
    # Set up the XGBoost classifier with initial parameters
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', objective='multi:softprob')
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model after grid search
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Train the best model with early stopping
    best_model.fit(X_train, y_train, 
                   eval_set=[(X_test, y_test)], 
                   early_stopping_rounds=10, 
                   verbose=False)
    
    # Predictions and probabilities
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print results
    print(f"Optimized XGBoost Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
  
  # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(best_model, importance_type='gain', max_num_features=10)
    plt.title("Top 10 Feature Importances")
    plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
    plt.close()

    # Return the trained model, predictions, and predicted probabilities
    return best_model, y_pred, y_scores
