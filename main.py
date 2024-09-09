from src.download import download_and_extract_dataset
from src.preprocessing import process_data
from src.visualization import visualze_data
def main():

    # Download the Heart Disease Dataset from Kaggle (save to data/csv/heart_disease_uci.csv)
    download_and_extract_dataset()

    # Gather our Data (import the csv)
    data = process_data('data/csv/heart_disease_uci.csv')

    # Visualize the Data (save into a folder called data/plots)
    

if __name__ == "__main__":
    main()
