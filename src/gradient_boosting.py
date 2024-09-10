from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

def train_and_evaluate(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Train and evaluate Gradient Boosting model."""
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Save results
    print(f"Gradient Boosting Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
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
