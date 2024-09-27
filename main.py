import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_precision_recall(models, X_test, y_test):
    plt.figure(figsize=(12, 8))  # Increased size for better clarity
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']  # Color list for variety
    color_idx = 0

    for model_name, model in models.items():
        if model is not None:
            try:
                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(X_test)
                else:
                    # Use decision_function for SVM if predict_proba is not available
                    y_scores = model.decision_function(X_test)
                    y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())  # Normalize

                for class_index in range(y_scores.shape[1]):
                    precision, recall, _ = precision_recall_curve(y_test == class_index, y_scores[:, class_index])
                    plt.plot(recall, precision, color=colors[color_idx % len(colors)], 
                             label=f'{model_name} - Class {class_index}')
                    color_idx += 1
            except Exception as e:
                print(f"Error while processing model {model_name}: {e}")

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision vs Recall for Multiple Models', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()  # Ensures that everything fits in the figure area
    plt.savefig('data/plots/precision_recall_plot.png')
    plt.show()  # Display the plot
    plt.close()  # Close the plot to free memory

