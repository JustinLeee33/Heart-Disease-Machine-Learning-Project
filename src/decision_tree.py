# decision_tree.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import label_binarize
from termcolor import colored  # Import for colored logging

def dt_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate Decision Tree model."""
    
    # Log the start of the process
    print(colored("\n--- Decision Tree Training Started ---", "green"))

    # Determine number of classes
    print(colored("Determining the number of classes...", "cyan"))
    n_classes = len(set(y_train))  # Number of unique classes in y_train

    # Initialize the model
    print(colored("Initializing the Decision Tree model...", "cyan"))
    model = DecisionTreeClassifier(random_state=42)
    
    # Train the model
    print(colored("Training the Decision Tree model...", "cyan"))
    model.fit(X_train, y_train)
    print(colored("Model training completed.", "green"))

    # Predict labels
    print(colored("Predicting labels for the test set...", "cyan"))
    y_pred = model.predict(X_test)
    print(colored("Prediction completed.", "green"))

    # Get predicted probabilities for all classes
    print(colored("Calculating predicted probabilities...", "cyan"))
    if hasattr(model, "predict_proba"):  # Check if the model supports predict_proba
        y_scores = model.predict_proba(X_test)  # Get probabilities for all classes
        print(colored("Predicted probabilities calculated.", "green"))
    else:
        # If the model doesn't support predict_proba, binarize the predictions for evaluation
        y_scores = label_binarize(y_pred, classes=range(n_classes))
        print(colored("Predicted probabilities binarized.", "yellow"))

    # Evaluation Metrics
    print(colored("\n--- Evaluating Model Performance ---", "blue"))
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')        # Use 'weighted' for multiclass

    # Print results
    print(colored(f"Decision Tree Accuracy: {accuracy:.4f}", "magenta"))
    print(colored("Classification Report:", "magenta"))
    print(report)
    print(colored(f"Confusion Matrix:\n{cm}", "magenta"))
    print(colored(f"Weighted Precision: {precision:.4f}", "magenta"))
    print(colored(f"Weighted Recall: {recall:.4f}", "magenta"))

    # Save classification report
    print(colored("\nSaving classification report...", "cyan"))
    with open(os.path.join(plot_dir, 'decision_tree_report.txt'), 'w') as f:
        f.write(report)
    print(colored("Classification report saved.", "green"))

    # Save accuracy plot
    print(colored("Saving accuracy plot...", "cyan"))
    plt.figure(figsize=(6, 4))
    plt.bar(['Decision Tree'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'decision_tree_accuracy.png'))
    plt.close()
    print(colored("Accuracy plot saved.", "green"))

    # Log the end of the process
    print(colored("\n--- Decision Tree Training and Evaluation Completed ---", "green"))

    # Return the model, predictions, and predicted probabilities (y_scores)
    return model, y_pred, y_scores
