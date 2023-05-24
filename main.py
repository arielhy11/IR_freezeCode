import pandas as pd
from sklearn.impute import SimpleImputer
from decisiontreemodel import DecisionTreeModel
from randomforest import RandomForestModel
from GradientBoostingModel import GradientBoostingModel
from naivebayes import NaiveBayesModel
from logisticregression import LogisticRegressionModel
from svm import SVMModel
from knn import KNNModel
from AdaBoostModel import AdaBoostModel
from ExtraTreeModel import ExtraTreeModel
import os

def main():
    # Load your csv file
    df = pd.read_csv('../numeric_two_first_plans_with_prediction.csv')

    # Apply SimpleImputer to handle missing data
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # List of models
    models = [
        DecisionTreeModel(df),
        RandomForestModel(df),
        GradientBoostingModel(df),
        NaiveBayesModel(df),
        LogisticRegressionModel(df),
        SVMModel(df),
        KNNModel(df),
        AdaBoostModel(df),
        ExtraTreeModel(df),
    ]

    # Check if the files exist and are not empty
    accuracy_file_exists = os.path.isfile('../conclusions/compare_algorithms.txt')
    accuracy_file_not_empty = accuracy_file_exists and os.path.getsize('../conclusions/compare_algorithms.txt') > 0

    feature_file_exists = os.path.isfile('../conclusions/feature_importances.txt')
    feature_file_not_empty = feature_file_exists and os.path.getsize('../conclusions/feature_importances.txt') > 0

    # Temporary files to store new results
    temp_accuracy_file = 'temp_compare_algorithms.txt'
    temp_feature_file = 'temp_feature_importances.txt'

    # Open the temporary files in write mode
    with open(temp_accuracy_file, 'w') as file, open(temp_feature_file, 'w') as feature_file:
        # Iterate over the models, train them, and write the results to the temporary files
        try:
            for model in models:
                results = model.fit_and_score()
                for hp, accuracy in results:
                    result_str = f'For {model.__class__.__name__} with hyperparameters {hp}, the accuracy is {accuracy}\n'
                    file.write(result_str)

                feature_importances = model.get_feature_importances()
                for feature_importance_str in feature_importances:
                    feature_file.write(feature_importance_str)

        except:
            # If an exception occurs, remove the temporary files and raise the exception
            os.remove(temp_accuracy_file)
            os.remove(temp_feature_file)
            raise

    # Replace the original files with the temporary files if they existed and were not empty before
    if accuracy_file_not_empty:
        os.replace(temp_accuracy_file, '../conclusions/compare_algorithms.txt')
    else:
        # If the original file was empty or did not exist, rename the temporary file to the original file
        os.rename(temp_accuracy_file, '../conclusions/compare_algorithms.txt')

    if feature_file_not_empty:
        if os.path.isfile('../conclusions/feature_importances.txt'):
            os.remove('../conclusions/feature_importances.txt')  # Remove the existing file
        os.rename(temp_feature_file, '../conclusions/feature_importances.txt')
    else:
        # If the original file was empty or did not exist, rename the temporary file to the original file
        os.rename(temp_feature_file, '../conclusions/feature_importances.txt')

if __name__ == "__main__":
    main()
