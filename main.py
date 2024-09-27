from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

def plot_precision_recall_curve(y_test, y_probs, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    avg_precision = average_precision_score(y_test, y_probs)

    plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')

def main():
    # Download and process dataset
    download_and_extract_dataset()
    data = process_data('data/csv/heart_disease_uci.csv')
    visualize_data(data)
    
    # Convert categorical data
    converted_data, category_mappings = convert_categorical_to_int(data)
    print("Category Mappings:\n", category_mappings)
    
    # Prepare data
    X = converted_data.drop(columns=['num'])
    y = converted_data['num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    models = [
        ('Logistic Regression', lr_train_and_evaluate(X_train, X_test, y_train, y_test)),
        ('Decision Tree', dt_train_and_evaluate(X_train, X_test, y_train, y_test)),
        ('Random Forest', rf_train_and_evaluate(X_train, X_test, y_train, y_test)),
        ('Gradient Boosting', gb_train_and_evaluate(X_train, X_test, y_train, y_test)),
        ('SVM', svm_train_and_evaluate(X_train, X_test, y_train, y_test))
    ]
    
    plt.figure(figsize=(10, 7))

    # Plot precision-recall curves for each model
    for model_name, (model, y_probs) in models:
        plot_precision_recall_curve(y_test, y_probs, model_name)
    
    plt.savefig('data/plots/precision_recall_curves.png')
    plt.show()

if __name__ == "__main__":
    main()
