# randomforest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

COLUMN_TO_PREDICT = 'is_long'

class RandomForestModel:
    def __init__(self, df):
        self.df = df
        self.y = df[COLUMN_TO_PREDICT]
        self.X = df.drop(COLUMN_TO_PREDICT, axis=1)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.hyperparameters = [
            {'n_estimators': 100, 'max_depth': 5},
            {'n_estimators': 200, 'max_depth': 10}
        ]
        self.models = []

    def fit_and_score(self):
        results = []
        for params in self.hyperparameters:
            n_estimators = params['n_estimators']
            max_depth = params['max_depth']
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled,
                self.y,
                test_size=0.2,
                random_state=42
            )
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results.append((params, accuracy))
            self.models.append(model)
        return results

    def get_feature_importances(self):
        feature_importance_strs = []
        for model, params in zip(self.models, self.hyperparameters):
            # Sort the feature importances and select the top three
            top_features = sorted(enumerate(model.feature_importances_), key=lambda x: x[1], reverse=True)[:3]

            feature_importance_str = f"For {self.__class__.__name__} with hyperparameters {params}, the top 3 features are: "
            for idx, importance in top_features:
                # Get the feature name
                feature_name = self.X.columns[idx]

                # Append the feature name and its importance to the output string
                feature_importance_str += f"{feature_name} ({importance:.2f}), "

            # Remove the last comma and space, and add a newline
            feature_importance_str = feature_importance_str[:-2] + "\n"

            feature_importance_strs.append(feature_importance_str)
        return feature_importance_strs

