import subprocess
import zipfile
import os
import shutil

def download_and_extract_dataset():
    print('Downloading the data')
    # Define the dataset file and directory
    dataset_file = 'heart-disease-data.zip'
    dataset_dir = 'heart-disease-data'
    destination_dir = 'data/csv/'

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Download the dataset from Kaggle
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'redwankarimsony/heart-disease-data', '--force'], check=True)

    # Unzip the downloaded dataset
    with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    # Move the extracted CSV to the destination directory
    extracted_file = os.path.join(dataset_dir, 'heart_disease_uci.csv')
    shutil.move(extracted_file, os.path.join(destination_dir, 'heart_disease_uci.csv'))

    print(f"Dataset downloaded and extracted to {destination_dir}")
