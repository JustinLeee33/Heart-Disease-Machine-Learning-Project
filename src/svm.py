from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import os

def svm_train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate Support Vector Machine model."""
    model = SVC(probability=True, random_state=42)  # Enable probability estimates
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate precision and recall for both binary and multiclass cases
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')        # Use 'weighted' for multiclass
    
    # Print results
    print(f"SVM Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    
    # Save classification report
    with open(os.path.join(plot_dir, 'svm_report.txt'), 'w') as f:
        f.write(report)
    
    # Save accuracy plot
    plt.figure(figsize=(6, 4))
    plt.bar(['SVM'], [accuracy])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'svm_accuracy.png'))
    plt.close()

    # Return the model and predictions to use for plotting or further evaluation
    return model, y_pred
