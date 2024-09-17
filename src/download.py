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
    kaggle_download_cmd = ['kaggle', 'datasets', 'download', '-d', 'redwankarimsony/heart-disease-data', '--force', '-p', '.']
    subprocess.run(kaggle_download_cmd, check=True)

    # Check if the zip file exists after download
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Download failed: {dataset_file} not found.")

    # Unzip the downloaded dataset
    with zipfile.ZipFile(dataset_file, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)

    # Move the extracted CSV to the destination directory
    extracted_file = os.path.join(dataset_dir, 'heart_disease_uci.csv')
    if not os.path.exists(extracted_file):
        raise FileNotFoundError(f"Extracted file not found: {extracted_file}")
    
    shutil.move(extracted_file, os.path.join(destination_dir, 'heart_disease_uci.csv'))

    print(f"Dataset downloaded and extracted to {destination_dir}")
