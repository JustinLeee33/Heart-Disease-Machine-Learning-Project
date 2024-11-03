from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
import os
from termcolor import colored  # Import for colored logging

def automl_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate an AutoML (TPOT) model with multiclass support."""
    
    # Log the start of the process
    print(colored("\n--- AutoML (TPOT) Training Started ---", "green"))

    # Initialize the model
    print(colored("Initializing the TPOT AutoML model...", "cyan"))
    model = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
    
    # Train the model
    print(colored("Training the TPOT AutoML model...", "cyan"))
    model.fit(X_train, y_train)
    print(colored("Model training completed.", "green"))

    # Predict labels
    print(colored("Predicting labels for the test set...", "cyan"))
    y_pred = model.predict(X_test)
    print(colored("Prediction completed.", "green"))

    # Get predicted probabilities for each class
    print(colored("Calculating predicted probabilities...", "cyan"))
    y_scores = model.predict_proba(X_test)  # y_scores will have shape (n_samples, n_classes)
    print(colored("Predicted probabilities calculated.", "green"))

    # Evaluation Metrics
    print(colored("\n--- Evaluating Model Performance ---", "blue"))
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Print results
    print(colored(f"AutoML (TPOT) Model Accuracy: {accuracy:.4f}", "magenta"))
    print(colored("Classification Report:", "magenta"))
    print(report)

    # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Save classification report
    print(colored("\nSaving classification report...", "cyan"))
    with open(os.path.join(plot_dir, 'automl_tpot_report.txt'), 'w') as f:
        f.write(report)
    print(colored("Classification report saved.", "green"))

    # Log the end of the process
    print(colored("\n--- AutoML (TPOT) Training and Evaluation Completed ---", "green"))

    # Return the fitted pipeline, predictions, and predicted probabilities
    return model.fitted_pipeline_, y_pred, y_scores
