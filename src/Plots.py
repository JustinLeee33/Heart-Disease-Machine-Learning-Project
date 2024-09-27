# In visualizations/plots.py
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import os

def save_plot(filename, plot_dir='data/plots'):
    """Helper function to save plots to the specified directory."""
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

def plot_precision_recall_curves(models, X_train, X_test, y_train, y_test, plot_dir='data/plots'):
    """Plots Precision vs Recall for all models."""
    plt.figure(figsize=(10, 6))

    for model_name, model_func in models.items():
        model = model_func(X_train, X_test, y_train, y_test, plot_dir)
        
        # Get the probability scores for the positive class
        y_scores = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        
        # Plot Precision vs Recall curve
        plt.plot(recall, precision, label=model_name)
    
    plt.title('Precision vs Recall for Different Models')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    save_plot('precision_recall_curves.png', plot_dir)
