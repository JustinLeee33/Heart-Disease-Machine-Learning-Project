import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(filename, plot_dir='data/plots'):
    """Helper function to save plots to the specified directory."""
    os.makedirs(plot_dir, exist_ok=True)  # Create the directory if it doesn't exist
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

def plot_missing_values(data, plot_dir='data/plots'):
    """Plots a bar chart showing missing values for each feature."""
    missing_values = data.isnull().sum()
    missing = missing_values[missing_values > 0]
    
    if not missing.empty:
        plt.figure(figsize=(10, 6))
        missing.plot(kind='bar', color='salmon')
        plt.title('Missing Values per Feature')
        plt.ylabel('Number of Missing Values')
        plt.tight_layout()
        save_plot('missing_values.png', plot_dir)

def plot_most_variable_columns(data, plot_dir='data/plots'):
    """Plots the most variable numerical columns based on standard deviation."""
    std_devs = data.std().sort_values(ascending=False)
    top_5_std = std_devs.head(5).index  # Top 5 most variable columns
    print(f"Most Variable Columns: {top_5_std.tolist()}")
    
    data[top_5_std].hist(figsize=(12, 8), bins=30, edgecolor='black')
    plt.suptitle('Histograms of Most Variable Features')
    plt.tight_layout()
    save_plot('most_variable_columns.png', plot_dir)

def plot_pairplot(data, plot_dir='data/plots'):
    """Plots a pairplot for visualizing relationships between features."""
    sns.pairplot(data)
    plt.suptitle('Pair Plot of Features', y=1.02)
    save_plot('pairplot.png', plot_dir)

def plot_correlation_heatmap(data, plot_dir='data/plots'):
    """Plots a heatmap of the correlation matrix."""
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    save_plot('correlation_heatmap.png', plot_dir)

def visualize_data(data):
    """Runs all visualization functions on the dataset and saves plots to data/plots."""
    plot_dir = 'data/plots'
    
    # Show missing values
    plot_missing_values(data, plot_dir)
    
    # Show the most variable columns
    plot_most_variable_columns(data, plot_dir)
    
    # Show correlation heatmap
    plot_correlation_heatmap(data, plot_dir)
    
    # Show pairplot to visualize relationships
    plot_pairplot(data, plot_dir)
