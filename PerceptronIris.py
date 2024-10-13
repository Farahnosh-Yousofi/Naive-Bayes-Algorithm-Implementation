import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Perceptron class for binary classification
class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self._activation(linear_output)
                update = self.learning_rate * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._activation(linear_output)
        return y_pred

    def _activation(self, x):
        return np.where(x >= 0, 1, 0)

# One-vs-Rest Perceptron for multi-class classification
class OneVsRestPerceptron:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.models = []

    def fit(self, X, y):
        n_classes = y.shape[1]
        for i in range(n_classes):
            perceptron = Perceptron(learning_rate=self.learning_rate, n_iters=self.n_iters)
            perceptron.fit(X, y[:, i])  # Train one classifier for each class
            self.models.append(perceptron)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return np.argmax(predictions, axis=1)

# Function to calculate and print evaluation metrics
def evaluate_model(y_true, y_pred, set_name=""):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Results for {set_name} Set:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%\n")

# Load the dataset
file_path = '~/Desktop/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(file_path, header=None, names=column_names)

# Prepare features and target
X = iris_df.drop('class', axis=1)
y = iris_df['class']

# Binarize the target for one-vs-rest classification
lb = LabelBinarizer()
y_binarized = lb.fit_transform(y)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_binarized, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the One-vs-Rest Perceptron
one_vs_rest_perceptron = OneVsRestPerceptron(learning_rate=0.1, n_iters=1000)
one_vs_rest_perceptron.fit(X_train.values, y_train)

# Predict on validation and test sets
val_predictions = one_vs_rest_perceptron.predict(X_val.values)
test_predictions = one_vs_rest_perceptron.predict(X_test.values)

# Convert one-hot encoded targets back to single labels
y_val_labels = np.argmax(y_val, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Evaluate on validation set
evaluate_model(y_val_labels, val_predictions, set_name="Validation")

# Evaluate on test set
evaluate_model(y_test_labels, test_predictions, set_name="Test")