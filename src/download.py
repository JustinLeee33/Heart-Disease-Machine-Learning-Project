import subprocess
import zipfile
import os

def download_and_extract_dataset():
    # Define the dataset file and directory
    dataset_file = 'heart-disease-data.zip'
    dataset_dir = 'heart-disease-data'

    # Download the dataset from Kaggle
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'redwankarimsony/heart-disease-data', '--force'], check=True)

    # Unzip the downloaded dataset
    with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    print(f"Dataset downloaded and extracted to {dataset_dir}")
