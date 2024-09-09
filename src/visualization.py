import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(filename):
    """Helper function to save plots to the specified directory."""
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

def summarize_data(data):
    """Prints summary statistics for each column in the DataFrame."""
    print("Summary Statistics:")
    for column in data.columns:
        print(f"\nColumn: {column}")
        print(f"Range: {data[column].min()} to {data[column].max()}")
        print(f"Number of items: {data[column].count()}")
        print(f"Mean: {data[column].mean()}")
        print(f"Standard Deviation: {data[column].std()}")

def plot_histograms(data):
    """Plots histograms for all numerical features."""
    data.hist(figsize=(12, 10), bins=30, edgecolor='black')
    plt.suptitle('Histograms of Numerical Features')
    plt.tight_layout()
    save_plot('histograms_numerical_features.png')

def plot_pairplot(data, target_column):
    """Plots a pairplot for visualizing relationships between features."""
    sns.pairplot(data, hue=target_column)
    plt.suptitle('Pair Plot of Features', y=1.02)
    save_plot('pair_plot_features.png')

def plot_correlation_heatmap(data):
    """Plots a heatmap of the correlation matrix."""
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    save_plot('correlation_heatmap.png')

def plot_boxplots(data, target_column):
    """Plots boxplots for numerical features against the target variable."""
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(14, 8))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(4, 4, i)  # Adjust the layout according to the number of features
        sns.boxplot(x=target_column, y=feature, data=data)
        plt.title(f'Boxplot of {feature}')
    plt.tight_layout()
    save_plot('boxplots_numerical_features.png')

def plot_countplot(data, target_column):
    """Plots a count plot for the target variable."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_column, data=data)
    plt.title('Count Plot of Target Variable')
    save_plot('count_plot_target_variable.png')

def visualize_data(data, target_column):
    """Runs all visualization functions on the dataset."""

    # Create the directory for saving plots if it does not exist
    plot_dir = 'data/plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Do the visualization
    summarize_data(data)
    plot_histograms(data)
    plot_pairplot(data, target_column)
    plot_correlation_heatmap(data)
    plot_boxplots(data, target_column)
    plot_countplot(data, target_column)
