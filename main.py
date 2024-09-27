from src.preprocessing import process_data, convert_categorical_to_int
from src.download import download_and_extract_dataset
from src.visualization import visualize_data
from src.logistic_regression import lr_train_and_evaluate
from src.decision_tree import dt_train_and_evaluate
from src.random_forest import rf_train_and_evaluate
from src.gradient_boosting import gb_train_and_evaluate
from src.svm import svm_train_and_evaluate

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import os

def plot_precision_recall(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        if model is not None:
            try:
                if hasattr(model, 'predict_proba'):
                    y_scores = model.predict_proba(X_test)
                else:
                    # Use decision_function for SVM if predict_proba is not available
                    y_scores = model.decision_function(X_test)
                
                for class_index in range(y_scores.shape[1]):
                    precision, recall, _ = precision_recall_curve(y_test == class_index, y_scores[:, class_index])
                    plt.plot(recall, precision, label=f'{model_name} - Class {class_index}')
            except Exception as e:
                print(f"Error while processing model {model_name}: {e}")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall for Multiple Models')
    plt.legend()
    plt.grid()
    plt.savefig('data/plots/precision_recall_plot.png')
    plt.show()  # Display the plot
    plt.close()  # Close the plot to free memory


def main():
    download_and_extract_dataset()
    data = process_data('data/csv/heart_disease_uci.csv')
    visualize_data(data)

    converted_data, category_mappings = convert_categorical_to_int(data)
    print("Category Mappings:\n", category_mappings)

    if 'num' in converted_data.columns:
        X = converted_data.drop(columns=['num'])
        y = converted_data['num']
    else:
        raise KeyError("The 'num' column (ground truth) is missing from the dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    os.makedirs('data/plots', exist_ok=True)

    models = {}
    models['Logistic Regression'] = lr_train_and_evaluate(X_train, X_test, y_train, y_test)
    models['Decision Tree'] = dt_train_and_evaluate(X_train, X_test, y_train, y_test)
    models['Random Forest'] = rf_train_and_evaluate(X_train, X_test, y_train, y_test)
    models['Gradient Boosting'] = gb_train_and_evaluate(X_train, X_test, y_train, y_test)
    models['SVM'] = svm_train_and_evaluate(X_train, X_test, y_train, y_test)

    plot_precision_recall(models, X_test, y_test)

if __name__ == "__main__":
    main()
