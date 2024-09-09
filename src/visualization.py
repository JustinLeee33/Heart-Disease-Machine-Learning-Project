import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(filename, plot_dir):
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

def plot_missing_values(data):
    """Plots a bar chart showing missing values for each feature."""
    missing_values = data.isnull().sum()
    missing = missing_values[missing_values > 0]
    
    if not missing.empty:
        plt.figure(figsize=(10, 6))
        missing.plot(kind='bar', color='salmon')
        plt.title('Missing Values per Feature')
        plt.ylabel('Number of Missing Values')
        plt.tight_layout()
        plt.show()

def plot_most_variable_columns(data):
    """Plots the most variable numerical columns based on standard deviation."""
    std_devs = data.std().sort_values(ascending=False)
    top_5_std = std_devs.head(5).index  # Top 5 most variable columns
    print(f"Most Variable Columns: {top_5_std.tolist()}")
    
    data[top_5_std].hist(figsize=(12, 8), bins=30, edgecolor='black')
    plt.suptitle('Histograms of Most Variable Features')
    plt.tight_layout()
    plt.show()

def plot_pairplot(data):
    """Plots a pairplot for visualizing relationships between features."""
    sns.pairplot(data)
    plt.suptitle('Pair Plot of Features', y=1.02)
    plt.show()

def plot_correlation_heatmap(data):
    """Plots a heatmap of the correlation matrix."""
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

def visualize_data(data):
    """Runs all visualization functions on the dataset without specifying a target column."""
    # Show missing values
    plot_missing_values(data)
    
    # Show the most variable columns
    plot_most_variable_columns(data)
    
    # Show correlation heatmap
    plot_correlation_heatmap(data)
    
    # Show pairplot to visualize relationships
    plot_pairplot(data)

# Example usage
file_path = 'data/csv/heart_disease_uci.csv'
data = pd.read_csv(file_path)

visualize_data(data)
