# logistic_regression.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import label_binarize
from termcolor import colored  # Import for colored logging

def lr_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate Logistic Regression model."""
    
    # Log the start of the process
    print(colored("\n--- Logistic Regression Training Started ---", "green"))

    # Initialize Logistic Regression model
    print(colored("Initializing the Logistic Regression model...", "cyan"))
    model = LogisticRegression(max_iter=1000, multi_class='ovr')

    # Train the model
    print(colored("Training the Logistic Regression model...", "cyan"))
    model.fit(X_train, y_train)
    print(colored("Model training completed.", "green"))

    # Predict labels
    print(colored("Predicting labels for the test set...", "cyan"))
    y_pred = model.predict(X_test)
    print(colored("Prediction completed.", "green"))

    # Get predicted probabilities for Precision-Recall curve
    print(colored("Calculating predicted probabilities...", "cyan"))
    y_scores = model.predict_proba(X_test)
    print(colored("Predicted probabilities calculated.", "green"))

    # Evaluation Metrics
    print(colored("\n--- Evaluating Model Performance ---", "blue"))
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')        # Use 'weighted' for multiclass

    # Print results
    print(colored(f"Logistic Regression Accuracy: {accuracy:.4f}", "magenta"))
    print(colored("Classification Report:", "magenta"))
    print(report)
    print(colored(f"Confusion Matrix:\n{cm}", "magenta"))
    print(colored(f"Weighted Precision: {precision:.4f}", "magenta"))
    print(colored(f"Weighted Recall: {recall:.4f}", "magenta"))

    # Save classification report
    print(colored("\nSaving classification report...", "cyan"))
    with open(os.path.join(plot_dir, 'logistic_regression_report.txt'), 'w') as f:
        f.write(report)
    print(colored("Classification report saved.", "green"))

    # Save accuracy plot
    print(colored("Saving accuracy plot...", "cyan"))
    plt.figure(figsize=(6, 4))
    plt.bar(['Logistic Regression'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'logistic_regression_accuracy.png'))
    plt.close()
    print(colored("Accuracy plot saved.", "green"))

    # Log the end of the process
    print(colored("\n--- Logistic Regression Training and Evaluation Completed ---", "green"))

    # Return the model, predictions, and predicted probabilities (y_scores)
    return model, y_pred, y_scores
