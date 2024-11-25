import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import os
from termcolor import colored
import warnings
import matplotlib.pyplot as plt

# Suppress specific XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Parameters: { \"use_label_encoder\" } are not used.*")

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
        eval_metric='mlogloss',
        objective='multi:softprob'
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

    # Include eval_set and eval_metric during training
    best_model.fit(X_train, y_train, eval_metric='mlogloss', eval_set=[(X_test, y_test)], verbose=True)  # verbose=True will print the progress
    print(colored("Model training completed.", "green"))

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

def plot_grid_search_results(param_values, accuracies):
    """Plot Grid Search results for visualizing hyperparameter tuning."""
    plt.plot(param_values, accuracies, marker='o')
    plt.title('Grid Search: Accuracy vs Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

# Example of running a Grid Search for max_depth hyperparameter values (if you want to visualize grid search results)
def grid_search_max_depth(X_train, y_train, X_test, y_test):
    param_grid = {'max_depth': [2, 3, 4, 5, 6]}
    grid_search = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_grid, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_max_depth = grid_search.best_params_['max_depth']
    print(f"Best max_depth from Grid Search: {best_max_depth}")

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for best max_depth ({best_max_depth}): {accuracy:.4f}")

    # Plot the results for max_depth grid search
    accuracies = grid_search.cv_results_['mean_test_score']
    plot_grid_search_results(param_values=param_grid['max_depth'], accuracies=accuracies)

# You can call grid_search_max_depth with your data to visualize the tuning of `max_depth`.
