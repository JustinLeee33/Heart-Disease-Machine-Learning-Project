# decision_tree.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import label_binarize

def dt_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate Decision Tree model."""
    # Determine number of classes
    n_classes = len(set(y_train))  # Number of unique classes in y_train

    # Initialize the model
    model = DecisionTreeClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict labels
    y_pred = model.predict(X_test)
    
    # Get predicted probabilities for all classes
    if hasattr(model, "predict_proba"):  # Check if the model supports predict_proba
        y_scores = model.predict_proba(X_test)  # Get probabilities for all classes
    else:
        # If the model doesn't support predict_proba, binarize the predictions for evaluation
        y_scores = label_binarize(y_pred, classes=range(n_classes))

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate precision and recall (weighted for multiclass cases)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')        # Use 'weighted' for multiclass
    
    # Print results
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    
    # Save classification report
    with open(os.path.join(plot_dir, 'decision_tree_report.txt'), 'w') as f:
        f.write(report)
    
    # Save accuracy plot
    plt.figure(figsize=(6, 4))
    plt.bar(['Decision Tree'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'decision_tree_accuracy.png'))
    plt.close()

    # Return the model, predictions, and predicted probabilities (y_scores)
    return model, y_pred, y_scores
