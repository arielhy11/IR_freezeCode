# NaiveBayesModel.py

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

COLUMN_TO_PREDICT = 'is_long'

class NaiveBayesModel:
    def __init__(self, df):
        self.df = df
        self.y = df[COLUMN_TO_PREDICT]
        self.X = df.drop(COLUMN_TO_PREDICT, axis=1)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def fit_and_score(self):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled,
            self.y,
            test_size=0.2,
            random_state=42
        )

        # Train the model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Make predictions and evaluate the accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # As there are no hyperparameters for GaussianNB, we just return the accuracy
        return [('N/A', accuracy)]

    def get_feature_importances(self):
        return []  # Naive Bayes does not provide feature importances
