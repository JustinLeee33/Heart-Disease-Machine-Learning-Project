from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Define the parameter grid
param_dist = {
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'reg_lambda': [0.1, 1, 10],
    'reg_alpha': [0, 0.1, 0.5],
    'n_estimators': [500, 1000, 1500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'gamma': [0, 0.1, 0.5],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Initialize the model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss')

# Initialize RandomizedSearchCV with the model and parameter grid
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings to sample
    scoring='accuracy',  # Use accuracy for scoring
    cv=3,  # 3-fold cross-validation
    verbose=1,  # Print progress
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters found: ", random_search.best_params_)
