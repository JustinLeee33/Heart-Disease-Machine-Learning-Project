from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_precision_recall_curves(X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Plots Precision vs Recall for all models."""
    models = {
        'Logistic Regression': lr_train_and_evaluate,
        'Decision Tree': dt_train_and_evaluate,
        'Random Forest': rf_train_and_evaluate,
        'Gradient Boosting': gb_train_and_evaluate,
        'SVM': svm_train_and_evaluate
    }
    
    plt.figure(figsize=(10, 6))
    
    for model_name, model_func in models.items():
        # Fit the model and get predictions
        model = model_func(X_train, X_test, y_train, y_test, plot_dir)
        
        # Calculate precision and recall
        y_scores = model.predict_proba(X_test)[:, 1]  # Get probability estimates for positive class
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        
        # Plot Precision vs Recall curve
        plt.plot(recall, precision, label=model_name)
    
    plt.title('Precision vs Recall for Different Models')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'precision_recall_curves.png'))
    plt.close()

# Update the main function to include the new plot
def main():
    # ... [rest of your existing code] ...
    
    # Train and evaluate models and plot P vs R curves
    plot_precision_recall_curves(X_train, X_test, y_train, y_test)

# Call the main function
if __name__ == '__main__':
    main()
