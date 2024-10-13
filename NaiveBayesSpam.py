import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from math import pi, exp, sqrt
from collections import defaultdict

# Text Preprocessing: Convert text into numeric features using CountVectorizer
def preprocess_text(dataset):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([item[1] for item in dataset])  # Transforming messages
    y = np.array([1 if item[0] == 'spam' else 0 for item in dataset])  # Converting labels
    return X.toarray(), y, vectorizer

# Step 1: Separate the dataset by class
def separate_by_class(dataset, labels):
    separated = defaultdict(list)
    for i, datapoint in enumerate(dataset):
        class_value = labels[i]
        separated[class_value].append(datapoint)
    return separated

# Step 2: Calculate the mean and standard deviation for a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / (len(numbers) - 1)
    return sqrt(variance)

# Step 3: Summarize the dataset (mean, stdev, count) for each feature
def summarize_dataset(dataset):
    summaries = [(mean(attribute), stdev(attribute), len(attribute)) for attribute in zip(*dataset)]
    return summaries

# Step 4: Summarize the dataset by class
def summarize_by_class(dataset, labels):
    separated = separate_by_class(dataset, labels)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize_dataset(instances)
    return summaries

# Step 5: Calculate Gaussian probability density function with epsilon for stability
def calculate_probability(x, mean, stdev):
    epsilon = 1e-9  # Small constant to avoid division by zero
    exponent = exp(-((x - mean) ** 2) / (2 * (stdev + epsilon) ** 2))
    return (1 / (sqrt(2 * pi) * (stdev + epsilon))) * exponent

# Step 6: Calculate class probabilities
def calculate_class_probabilities(summaries, input_vector):
    total_rows = sum([summaries[class_value][0][2] for class_value in summaries])
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)  # Prior probability
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(input_vector[i], mean, stdev)
    return probabilities

# Step 7: Make a prediction for a new input
def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

# Step 8: Naive Bayes training and prediction
def naive_bayes(train, train_labels, test):
    summaries = summarize_by_class(train, train_labels)
    predictions = []
    for row in test:
        output = predict(summaries, row)
        predictions.append(output)
    return predictions

# Step 9: Evaluate the model using accuracy, precision, recall, and F1-score
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Step 10: K-Fold Cross Validation with k=5
def k_fold_cross_validation(dataset, labels, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Store the results for each fold
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

    for train_index, test_index in kf.split(dataset):
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Train and predict using Naive Bayes
        y_pred = naive_bayes(X_train, y_train, X_test)

        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

        # Append the results for this fold
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Calculate average metrics across all folds
    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)

    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

# Main function to run the model on the SMS Spam dataset
def run_naive_bayes_spam_classifier(dataset):
    # Preprocess the dataset: convert text to numeric form
    X, y, vectorizer = preprocess_text(dataset)

    # Perform 5-fold cross-validation
    k_fold_cross_validation(X, y, k=5)

# Load your dataset into a pandas DataFrame
file_path = "SMSSpamCollection"
with open(file_path, 'r') as file:
    dataset_lines = file.readlines()

# Convert to pandas DataFrame
data = [line.strip().split('\t') for line in dataset_lines]
df = pd.DataFrame(data, columns=['label', 'message'])

# Prepare the dataset
spam_dataset = list(zip(df['label'], df['message']))

# Run the classifier with k-fold validation
run_naive_bayes_spam_classifier(spam_dataset)