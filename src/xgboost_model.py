from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os

def tune_and_plot_xgb(X_train, X_test, y_train, y_test, n_estimators_list, max_depth_list, plot_dir='data/plots'):
    """
    Train and evaluate XGBoost with varying n_estimators and max_depth,
    and create plots to visualize performance.
    """
    print(colored("\n--- Hyperparameter Tuning and Visualization Started ---", "green"))
    
    # Ensure plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize results storage
    results = []
    
    # Loop through parameter combinations
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            print(colored(f"Training model with n_estimators={n_estimators}, max_depth={max_depth}", "cyan"))
            
            # Initialize and train the model
            model = xgb.XGBClassifier(
                subsample=0.8,
                reg_lambda=1,
                reg_alpha=0,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.01,
                gamma=0,
                colsample_bytree=0.8,
                objective='multi:softmax',
                eval_metric='mlogloss'
            )
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store the results
            results.append({'n_estimators': n_estimators, 'max_depth': max_depth, 'accuracy': accuracy})
            print(colored(f"Accuracy: {accuracy:.4f}", "magenta"))
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot accuracy vs n_estimators for each max_depth
    for max_depth in max_depth_list:
        subset = results_df[results_df['max_depth'] == max_depth]
        plt.figure()
        plt.plot(subset['n_estimators'], subset['accuracy'], marker='o')
        plt.title(f'Accuracy vs n_estimators (max_depth={max_depth})')
        plt.xlabel('n_estimators')
        plt.ylabel('Accuracy')
        plt.grid()
        plot_path = os.path.join(plot_dir, f'accuracy_vs_n_estimators_max_depth_{max_depth}.png')
        plt.savefig(plot_path)
        print(colored(f"Plot saved to {plot_path}", "green"))
        plt.close()
    
    # Plot accuracy vs max_depth for each n_estimators
    for n_estimators in n_estimators_list:
        subset = results_df[results_df['n_estimators'] == n_estimators]
        plt.figure()
        plt.plot(subset['max_depth'], subset['accuracy'], marker='o')
        plt.title(f'Accuracy vs max_depth (n_estimators={n_estimators})')
        plt.xlabel('max_depth')
        plt.ylabel('Accuracy')
        plt.grid()
        plot_path = os.path.join(plot_dir, f'accuracy_vs_max_depth_n_estimators_{n_estimators}.png')
        plt.savefig(plot_path)
        print(colored(f"Plot saved to {plot_path}", "green"))
        plt.close()
    
    # Save the results to a CSV file
    results_csv_path = os.path.join(plot_dir, 'xgb_hyperparameter_tuning_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(colored(f"Results saved to {results_csv_path}", "green"))
    
    print(colored("\n--- Hyperparameter Tuning and Visualization Completed ---", "green"))

def xgb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate XGBoost model with hyperparameter tuning."""
    
    # Log the start of the process
    print(colored("\n--- XGBoost Training and Evaluation Started ---", "green"))
    
    # Initialize the model with the best parameters from RandomizedSearchCV
    print(colored("Initializing the XGBoost model...", "cyan"))
    best_model = xgb.XGBClassifier(
        subsample=0.8,
        reg_lambda=1,
        reg_alpha=0,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        gamma=0,
        colsample_bytree=0.8,
        objective='multi:softmax',  # For multi-class classification
        eval_metric='mlogloss'
    )
    
    # Train the best model
    print(colored("Training the best model...", "cyan"))
    best_model.fit(X_train, y_train)
    print(colored("Model training completed.", "green"))
    
    # Predict and evaluate
    print(colored("Predicting and evaluating model performance...", "cyan"))
    y_pred = best_model.predict(X_test)  # Get predicted class labels
    
    # No need for np.argmax if it's already a class label
    y_scores = best_model.predict_proba(X_test)  # Get predicted probabilities for evaluation
    
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
