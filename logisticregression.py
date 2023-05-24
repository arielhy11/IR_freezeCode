# LogisticRegressionModel.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

COLUMN_TO_PREDICT = 'is_long'

class LogisticRegressionModel:
    def __init__(self, df):
        self.df = df
        self.y = df[COLUMN_TO_PREDICT]
        self.X = df.drop(COLUMN_TO_PREDICT, axis=1)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.hyperparameters = [
            {'penalty': 'l1', 'solver': 'liblinear'},
            {'penalty': 'l2', 'solver': 'liblinear'}
        ]
        self.models = []

    def fit_and_score(self):
        results = []
        for params in self.hyperparameters:
            penalty = params['penalty']
            solver = params['solver']
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled,
                self.y,
                test_size=0.2,
                random_state=42
            )
            model = LogisticRegression(penalty=penalty, solver=solver, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results.append((params, accuracy))
            self.models.append(model)
        return results

    def get_feature_importances(self):
        return []  # Logistic Regression does not provide feature importances
