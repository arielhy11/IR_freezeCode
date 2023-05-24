import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

COLUMN_TO_PREDICT = "is_long"
FILE_TO_PREDICT = "../numeric_two_first_plans_with_prediction.csv"

class MLPModel:
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path)
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.X = self.data.drop(COLUMN_TO_PREDICT, axis=1)
        self.X_scaled = self.scaler.fit_transform(self.imputer.fit_transform(self.X))
        self.y = self.data[COLUMN_TO_PREDICT]
        self.model = MLPClassifier()
        self.best_params = {}
        self.best_accuracy = 0.0

    def fit(self):
        # Define the range of hyperparameter values to explore
        hyperparams = {
            'hidden_layer_sizes': [(50,), (100,), (200,)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01]
            # Add more hyperparameters and their corresponding ranges here
        }

        # Iterate through different hyperparameter options
        for i in range(3):
            hidden_layer_sizes = hyperparams['hidden_layer_sizes'][i]
            activation = hyperparams['activation'][i]
            alpha = hyperparams['alpha'][i]

            self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha)
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.transform(self.imputer.transform(X_train))
            X_val_scaled = self.scaler.transform(self.imputer.transform(X_val))
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)

            # Update best parameters and accuracy if current accuracy is higher
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_params['hidden_layer_sizes'] = hidden_layer_sizes
                self.best_params['activation'] = activation
                self.best_params['alpha'] = alpha

    def predict(self, X_new):
        X_new_scaled = self.scaler.transform(self.imputer.transform(X_new))
        y_pred = self.model.predict(X_new_scaled)
        return y_pred

    def score(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(self.imputer.transform(X_test))
        accuracy = self.model.score(X_test_scaled, y_test)
        return accuracy

# Load CSV file into a pandas DataFrame
df = pd.read_csv(FILE_TO_PREDICT)

# Initialize the MLPModel object
model = MLPModel(FILE_TO_PREDICT)

# Fit the model and find the best hyperparameters
model.fit()

# Print the best hyperparameters and accuracy
print('Best Hyperparameters:')
for param, value in model.best_params.items():
    print(f"{param}: {value}")
print('Best Accuracy:', model.best_accuracy)
