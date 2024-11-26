from tqdm import tqdm  # Progress bar for monitoring
from termcolor import colored
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing import process_data, convert_categorical_to_int
from src.download import download_and_extract_dataset
from src.visualization import visualize_data
from src.xgboost_model import xgb_train_and_evaluate
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def main():
    # Ensure the data/plots directory exists
    os.makedirs('data/plots', exist_ok=True)

    print(colored("\n--- Starting Data Preparation ---", "green"))

    # Download and process the dataset
    print(colored("Downloading and extracting the dataset...", "cyan"))
    download_and_extract_dataset()
    print(colored("Dataset downloaded and extracted successfully.", "green"))

    print(colored("Processing data...", "cyan"))
    data = process_data('data/csv/heart_disease_uci.csv')
    visualize_data(data)
    print(colored("Data processed successfully.", "green"))

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

    print(colored("Normalizing the features...", "cyan"))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    print(colored("Features normalized successfully.", "green"))

    print(colored("Splitting data into training and testing sets...", "cyan"))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(colored("Data split successfully.", "green"))

    # Define hyperparameter ranges
    n_estimators_range = range(5, 501, 50)
    max_depth_range = range(1, 26)

    results = []

    # Monitor progress
    total_iterations = len(n_estimators_range) * len(max_depth_range)
    with tqdm(total=total_iterations, desc="XGBoost Training") as pbar:
        for n_estimators in n_estimators_range:
            for max_depth in max_depth_range:
                # Train and evaluate the model
                model = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    subsample=0.8,
                    learning_rate=0.01,
                    colsample_bytree=0.8,
                    objective='multi:softmax',
                    eval_metric='mlogloss'
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Store the results
                results.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'accuracy': accuracy
                })

                # Update progress bar
                pbar.update(1)

    # Save results to a CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join('data/plots', 'xgb_hyperparameter_tuning_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(colored(f"Results saved to {results_csv_path}", "green"))

    # Create plots
    plot_results(results_df, 'data/plots')


def plot_results(results_df, plot_dir):
    """Plot accuracy for varying n_estimators and max_depth."""
    os.makedirs(plot_dir, exist_ok=True)

    # Plot accuracy vs. n_estimators for each max_depth
    for max_depth in results_df['max_depth'].unique():
        subset = results_df[results_df['max_depth'] == max_depth]
        plt.figure()
        plt.plot(subset['n_estimators'], subset['accuracy'], marker='o')
        plt.title(f'Accuracy vs n_estimators (max_depth={max_depth})')
        plt.xlabel('n_estimators')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plot_path = os.path.join(plot_dir, f'accuracy_vs_n_estimators_max_depth_{max_depth}.png')
        plt.savefig(plot_path)
        plt.close()

    # Plot accuracy vs. max_depth for each n_estimators
    for n_estimators in results_df['n_estimators'].unique():
        subset = results_df[results_df['n_estimators'] == n_estimators]
        plt.figure()
        plt.plot(subset['max_depth'], subset['accuracy'], marker='o')
        plt.title(f'Accuracy vs max_depth (n_estimators={n_estimators})')
        plt.xlabel('max_depth')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plot_path = os.path.join(plot_dir, f'accuracy_vs_max_depth_n_estimators_{n_estimators}.png')
        plt.savefig(plot_path)
        plt.close()


if __name__ == '__main__':
    main()
