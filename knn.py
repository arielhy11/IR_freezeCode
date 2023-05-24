# KNNModel.py

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

COLUMN_TO_PREDICT = 'is_long'

class KNNModel:
    def __init__(self, df):
        self.df = df
        self.y = df[COLUMN_TO_PREDICT]
        self.X = df.drop(COLUMN_TO_PREDICT, axis=1)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.hyperparameters = [3, 5]  # Define hyperparameters within the class

    def fit_and_score(self):
        results = []
        for n_neighbors in self.hyperparameters:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled,
                self.y,
                test_size=0.2,
                random_state=42
            )
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results.append((n_neighbors, accuracy))
        return results

    def get_feature_importances(self):
        return []  # KNN does not have built-in feature importances
