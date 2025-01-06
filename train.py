from src.preprocessing import process_data, convert_categorical_to_int
from src.download import download_and_extract_dataset
from src.visualization import visualize_data
from src.logistic_regression import lr_train_and_evaluate
from src.decision_tree import dt_train_and_evaluate
from src.random_forest import rf_train_and_evaluate
from src.gradient_boosting import gb_train_and_evaluate
from src.svm import svm_train_and_evaluate
from src.xgboost_model import xgb_train_and_evaluate
from src.automl import automl_train_and_evaluate

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import label_binarize, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import os
from termcolor import colored  # For colored logging

def main():
    # Ensure the data/plots directory exists
    os.makedirs('data/plots', exist_ok=True)

    print(colored("\n--- Starting Data Preparation ---", "green"))

    # Download the Heart Disease Dataset from Kaggle
    print(colored("Downloading and extracting the dataset...", "cyan"))
    download_and_extract_dataset()
    print(colored("Dataset downloaded and extracted successfully.", "green"))

    # Import and process the data
    print(colored("Processing data...", "cyan"))
    data = process_data('data/csv/heart_disease_uci.csv')
    visualize_data(data)
    print(colored("Data processed successfully.", "green"))

    # Convert categorical columns to integers
    print(colored("Converting categorical columns to integer mappings...", "cyan"))
    converted_data, category_mappings = convert_categorical_to_int(data)
    print("Category Mappings:\n", category_mappings)
    print(colored("Categorical data converted successfully.", "green"))

    # Separate features and target
    if 'num' in converted_data.columns:
        X = converted_data.drop(columns=['num'])
        y = converted_data['num']
    else:
        raise KeyError(colored("The 'num' column (ground truth) is missing from the dataset.", "red"))

    # Define the scaler
    scaler = MinMaxScaler()

    # Normalize the features
    print(colored("Normalizing the features...", "cyan"))
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    print(colored("Features normalized successfully.", "green"))

    # Determine the number of classes
    n_classes = len(y.unique())

    # Split data
    print(colored("Splitting data into training and testing sets...", "cyan"))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(colored("Data split successfully.", "green"))

    # Define models
    models = {
        #'Logistic Regression': lr_train_and_evaluate,
        #'Decision Tree': dt_train_and_evaluate,
        #'Random Forest': rf_train_and_evaluate,
        #'Gradient Boosting': gb_train_and_evaluate,
        #'SVM': svm_train_and_evaluate,
        'XGBoost': xgb_train_and_evaluate,
        'XGBoost': tune_and_plot_xgb
        #'AutoML (TPOT)': automl_train_and_evaluate,
    }

    # Store scores for plotting
    models_scores = {}

    # Train and evaluate models
    for model_name, model_func in models.items():
        print(colored(f"\n--- Training and Evaluating {model_name} ---", "blue"))
        model, y_pred, y_scores = model_func(
            X_train, X_test, y_train, y_test, plot_dir='data/plots'
        )
        print(colored(f"{model_name} training and evaluation completed.", "green"))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(colored(f"{model_name} Confusion Matrix:\n{cm}", "magenta"))

        # Precision and recall (weighted average for multiclass)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Print results
        print(colored(f"{model_name} Weighted Precision: {precision:.2f}", "magenta"))
        print(colored(f"{model_name} Weighted Recall: {recall:.2f}\n", "magenta"))

        # Store scores (ensure y_scores is in the correct shape)
        if y_scores.ndim == 1:
            y_scores = label_binarize(y_scores, classes=range(n_classes))
        models_scores[model_name] = y_scores

    # Plot Precision vs Recall curves
    plot_precision_recall_curves(y_test, models_scores, n_classes, plot_dir='data/plots')

    # Compare reports
    print(colored("\n--- Comparing Model Reports ---", "green"))
    compare_model_reports('data/plots')

def plot_precision_recall_curves(y_test, models_scores, n_classes, plot_dir='data/plots'):
    """Plots Precision vs Recall for all models in multiclass setting."""
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))

    plt.figure(figsize=(10, 6))
    for model_name, y_scores in models_scores.items():
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_scores[:, i])
            average_precision[i] = average_precision_score(y_test_binarized[:, i], y_scores[:, i])
            plt.plot(recall[i], precision[i], lw=2, label=f'{model_name} class {i} (AP={average_precision[i]:0.2f})')

    plt.title('Precision-Recall Curves for Different Models and Classes')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'precision_recall_curves_multiclass.png'))
    plt.close()

def compare_model_reports(report_dir):
    """Compare and display the metrics from report .txt files."""
    report_files = [f for f in os.listdir(report_dir) if f.endswith('_report.txt')]
    report_summaries = {}

    for report_file in report_files:
        with open(os.path.join(report_dir, report_file), 'r') as f:
            report_content = f.read()
            # Parse out key metrics like precision, recall, and f1-score
            report_summaries[report_file] = report_content

    for report_name, report_content in report_summaries.items():
        print(colored(f"\n--- {report_name} ---", "cyan"))
        print(report_content)

if __name__ == '__main__':
    main()
