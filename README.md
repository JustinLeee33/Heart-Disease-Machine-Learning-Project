
# Heart Disease Machine Learning Project

## Overview
This repository is dedicated to building and evaluating machine learning models to predict heart disease outcomes based on medical data. The project encompasses various data processing, modeling, and evaluation strategies, leveraging popular machine learning frameworks and Python libraries.

View /Assets for the full research paper and presentation.

## Project Structure
The repository is structured as follows:

- **train.py**: The main script to run the data processing pipeline, model training, and evaluation.
- **src/**: Contains all the supporting modules for data preprocessing, model training, and visualization.
    - **src/preprocessing.py**: Functions for data cleaning, handling missing values, and converting categorical data to integer mappings.
    - **src/download.py**: A utility to download and extract the heart disease dataset from Kaggle.
    - **src/visualization.py**: Functions to generate visual plots and gain insights into the dataset.
    - **src/logistic_regression.py**: Implements the logistic regression model training and evaluation.
    - **src/decision_tree.py**: Code to train and evaluate a decision tree classifier.
    - **src/random_forest.py**: Code for training and evaluating a random forest model.
    - **src/gradient_boosting.py**: Gradient boosting model training and evaluation.
    - **src/svm.py**: Support Vector Machine (SVM) training and evaluation.
    - **src/xgboost_model.py**: Implements training and evaluation for an XGBoost model.
    - **src/automl.py**: Integrates an AutoML approach using TPOT to automatically find the best model configuration.
- **requirements.txt**: Lists all the dependencies required for running the project.
- **README.md**: Provides an overview and detailed information about the project.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Heart-Disease-Machine-Learning-Project.git
   cd Heart-Disease-Machine-Learning-Project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that you have access to Kaggle's API credentials to download the dataset.

## How to Run the Project
1. **Download and Prepare Data**:
   Ensure that your Kaggle credentials are set up properly. Run the main training script:
   ```bash
   python train.py
   ```

2. **Train Models**:
   The script will process the data, train multiple models, and evaluate their performance, including:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - SVM
   - XGBoost
   - AutoML using TPOT

3. **View Results**:
   Evaluation results, including confusion matrices and precision-recall curves, are stored in the `data/plots/` directory.

## Key Features
- **Comprehensive Preprocessing**: Handles missing data, encodes categorical variables, and scales numerical features.
- **Multiple Model Support**: Trains various classifiers and compares their performance.
- **AutoML Integration**: TPOT is used to automate the search for the best model and hyperparameters.
- **Visualization**: Precision-recall curves and confusion matrices are plotted for detailed evaluation.
- **Category Mapping**: The project maps categorical features to integers for seamless model input, and prints mappings for traceability.

## Data Preparation
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data). It includes patient data with various features related to cardiovascular health. Missing values are handled using mean imputation for numerical columns.

## Example Output
**Sample Category Mappings**:
```python
Category Mappings:
{
    'sex': {0: 'Male', 1: 'Female'},
    'cp': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'},
    ...
}
```

## Contributing
Contributions are welcome! Please create an issue or submit a pull request for any feature enhancements or bug fixes.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any questions or feedback, please contact justinleethirtythree@gmail.com or (803) 944-7922

---

**Note**: Ensure you handle the data according to Kaggle's and the dataset owner's licensing terms.
