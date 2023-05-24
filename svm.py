# SVMModel.py

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

COLUMN_TO_PREDICT = 'is_long'

class SVMModel:
    def __init__(self, df):
        self.df = df.sample(frac=0.1, random_state=1)  # Use 10% of the data
        self.y = self.df[COLUMN_TO_PREDICT]
        self.X = self.df.drop(COLUMN_TO_PREDICT, axis=1)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.hyperparameters = [
            {'C': 1, 'kernel': 'linear'},
            {'C': 0.5, 'kernel': 'linear'}
        ]
        self.models = []

    def fit_and_score(self):
        results = []
        for params in self.hyperparameters:
            C = params['C']
            kernel = params['kernel']
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled,
                self.y,
                test_size=0.2,
                random_state=42
            )
            model = SVC(C=C, kernel=kernel)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results.append((params, accuracy))
            self.models.append(model)
        return results

    def get_feature_importances(self):
        return []  # SVM does not provide feature importances
