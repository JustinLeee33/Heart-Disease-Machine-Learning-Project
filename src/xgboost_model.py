import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, classification_report
import os
from termcolor import colored  # Import for color logging

# Load in our preprocessed dataset
data = pd.read_csv('medical_costs_preprocessed.csv')

# Define features (X) and target (y)
X = data.drop(columns=['charges'])  # Assuming 'charges' is the target variable
y = data['charges']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning."""
    
    # Log the start of the process
    print(colored("\n--- XGBoost Training and Evaluation Started ---", "green"))
    
    # Define hyperparameters to explore
    print(colored("Defining hyperparameter grid for tuning...", "cyan"))
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1, 1.5]
    }

    # Initialize the model
    print(colored("Initializing the XGBoost model...", "cyan"))
    model = xgb.XGBRegressor(
        eval_metric='mae',  # Mean Absolute Error for regression tasks
        objective='reg:squarederror'  # Suitable for regression
    )

    # Hyperparameter tuning with Randomized Search
    print(colored("Starting Randomized Search for hyperparameter tuning...", "cyan"))
    randomized_search = RandomizedSearchCV(
        model, param_distributions=param_grid, scoring='neg_mean_absolute_error', 
        cv=3, n_jobs=-1, n_iter=100, verbose=1, random_state=42
    )
    randomized_search.fit(X_train, y_train)
    print(colored("Randomized Search completed. Best hyperparameters found.", "green"))

    # Train the best model
    print(colored("Training the best model...", "cyan"))
    best_model = randomized_search.best_estimator_
    
    # Train with early stopping
    best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    print(colored("Model training completed.", "green"))

    # Predict and evaluate
    print(colored("Predicting and evaluating model performance...", "cyan"))
    y_pred = best_model.predict(X_test)
    
    # Calculate and print metrics
    mae = mean_absolute_error(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(colored(f"Optimized XGBoost Mean Absolute Error: {mae:.4f}", "magenta"))
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
    
    # Return the model and predictions
    return best_model, y_pred

# Run the function with your dataset
model, predictions = xgb_train_and_evaluate(X_train, X_test, y_train, y_test)
