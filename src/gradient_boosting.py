# gradient_boosting.py

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import label_binarize

def gb_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate Gradient Boosting model."""
    # Instantiate Gradient Boosting Classifier
    model = GradientBoostingClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Predict labels
    y_pred = model.predict(X_test)
    
    # Determine the number of classes
    n_classes = len(set(y_train))

    # Get predicted probabilities for each class for Precision-Recall curve
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)  # For multiclass, y_scores will be a 2D array
    else:
        # If predict_proba is not available, binarize y_pred as a fallback
        y_scores = label_binarize(y_pred, classes=range(n_classes))
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate precision and recall for multiclass case
    precision = precision_score(y_test, y_pred, average='weighted')  # Weighted for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')        # Weighted for multiclass
    
    # Print results
    print(f"Gradient Boosting Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    
    # Save classification report
    with open(os.path.join(plot_dir, 'gradient_boosting_report.txt'), 'w') as f:
        f.write(report)
    
    # Save accuracy plot
    plt.figure(figsize=(6, 4))
    plt.bar(['Gradient Boosting'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'gradient_boosting_accuracy.png'))
    plt.close()

    # Return the model, predictions, and predicted probabilities (y_scores)
    return model, y_pred, y_scores
