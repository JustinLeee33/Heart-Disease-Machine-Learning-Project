# svm.py

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import label_binarize
from termcolor import colored  # Import for colored logging

def svm_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate Support Vector Machine model for multiclass classification."""

    # Log the start of the process
    print(colored("\n--- SVM Training Started ---", "green"))

    # Initialize the model with probability enabled
    print(colored("Initializing the SVM model with probability estimates...", "cyan"))
    model = SVC(probability=True, random_state=42)

    # Train the model
    print(colored("Training the SVM model...", "cyan"))
    model.fit(X_train, y_train)
    print(colored("Model training completed.", "green"))

    # Predict labels
    print(colored("Predicting labels for the test set...", "cyan"))
    y_pred = model.predict(X_test)
    print(colored("Prediction completed.", "green"))

    # Get predicted probabilities for all classes
    print(colored("Calculating predicted probabilities...", "cyan"))
    y_scores = model.predict_proba(X_test)  # Probability estimates for all classes
    print(colored("Predicted probabilities calculated.", "green"))

    # Evaluation Metrics
    print(colored("\n--- Evaluating Model Performance ---", "blue"))
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')        # Use 'weighted' for multiclass

    # Print results
    print(colored(f"SVM Accuracy: {accuracy:.4f}", "magenta"))
    print(colored("Classification Report:", "magenta"))
    print(report)
    print(colored(f"Confusion Matrix:\n{cm}", "magenta"))
    print(colored(f"Weighted Precision: {precision:.4f}", "magenta"))
    print(colored(f"Weighted Recall: {recall:.4f}", "magenta"))

    # Save classification report
    print(colored("\nSaving classification report...", "cyan"))
    with open(os.path.join(plot_dir, 'svm_report.txt'), 'w') as f:
        f.write(report)
    print(colored("Classification report saved.", "green"))

    # Save accuracy plot
    print(colored("Saving accuracy plot...", "cyan"))
    plt.figure(figsize=(6, 4))
    plt.bar(['SVM'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'svm_accuracy.png'))
    plt.close()
    print(colored("Accuracy plot saved.", "green"))

    # Log the end of the process
    print(colored("\n--- SVM Training and Evaluation Completed ---", "green"))

    # Return the model, predictions, and predicted probabilities for all classes (y_scores)
    return model, y_pred, y_scores
