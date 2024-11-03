# random_forest.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from termcolor import colored  # Import for colored logging

def rf_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots', n_classes=5):
    """Train and evaluate Random Forest model."""
    
    # Log the start of the process
    print(colored("\n--- Random Forest Training Started ---", "green"))

    # Initialize the model
    print(colored("Initializing the Random Forest model...", "cyan"))
    model = RandomForestClassifier(random_state=42)

    # Train the model
    print(colored("Training the Random Forest model...", "cyan"))
    model.fit(X_train, y_train)
    print(colored("Model training completed.", "green"))

    # Predict labels
    print(colored("Predicting labels for the test set...", "cyan"))
    y_pred = model.predict(X_test)
    print(colored("Prediction completed.", "green"))

    # Get predicted probabilities for all classes (multiclass)
    print(colored("Calculating predicted probabilities...", "cyan"))
    y_scores = model.predict_proba(X_test)  # Probabilities for each class
    print(colored("Predicted probabilities calculated.", "green"))

    # Evaluation Metrics
    print(colored("\n--- Evaluating Model Performance ---", "blue"))
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')        # Use 'weighted' for multiclass

    # Print results
    print(colored(f"Random Forest Accuracy: {accuracy:.4f}", "magenta"))
    print(colored("Classification Report:", "magenta"))
    print(report)
    print(colored(f"Confusion Matrix:\n{cm}", "magenta"))
    print(colored(f"Weighted Precision: {precision:.4f}", "magenta"))
    print(colored(f"Weighted Recall: {recall:.4f}", "magenta"))

    # Save classification report
    print(colored("\nSaving classification report...", "cyan"))
    report_path = os.path.join(plot_dir, 'random_forest_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(colored(f"Classification report saved to {report_path}.", "green"))

    # Save accuracy plot
    print(colored("Saving accuracy plot...", "cyan"))
    plt.figure(figsize=(6, 4))
    plt.bar(['Random Forest'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    accuracy_plot_path = os.path.join(plot_dir, 'random_forest_accuracy.png')
    plt.savefig(accuracy_plot_path)
    plt.close()
    print(colored(f"Accuracy plot saved to {accuracy_plot_path}.", "green"))

    # Log the end of the process
    print(colored("\n--- Random Forest Training and Evaluation Completed ---", "green"))

    # Return the model, predictions, and predicted probabilities (y_scores)
    return model, y_pred, y_scores
