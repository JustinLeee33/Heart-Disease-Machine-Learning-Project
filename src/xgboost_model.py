from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import numpy as np

# Define the hyperparameters grid
param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.1, 0.5, 1.0]
}

# Create an XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss')

# Use RandomizedSearchCV for tuning
random_search = RandomizedSearchCV(
    xgb_model, 
    param_distributions=param_dist, 
    n_iter=50, # Number of iterations
    cv=3, # Cross-validation splitting strategy
    verbose=2, # Show progress
    random_state=42, # For reproducibility
    n_jobs=-1 # Use all available CPUs
)

# Fit the model to the training data
random_search.fit(X_train, y_train)

# Get the best model and hyperparameters
best_model = random_search.best_estimator_
print("Best Hyperparameters:", random_search.best_params_)

# Optionally, evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of best model: {accuracy:.4f}")
