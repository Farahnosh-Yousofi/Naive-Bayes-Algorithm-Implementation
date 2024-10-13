from math import pi, exp, sqrt
from collections import defaultdict

# Step 1: Separate the dataset by class
def separate_by_class(dataset):
    separated = defaultdict(list)
    for datapoint in dataset:
        class_value = datapoint[-1]
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
    del summaries[-1]  # Remove the class label column
    return summaries

# Step 4: Summarize the dataset by class
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize_dataset(instances)
    return summaries

# Step 5: Calculate Gaussian probability density function
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2) / (2 * stdev ** 2))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

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

# Step 8: Test the Naive Bayes algorithm
def naive_bayes(train, test):
    summaries = summarize_by_class(train)
    predictions = []
    for row in test:
        output = predict(summaries, row)
        predictions.append(output)
    return predictions

# Step 9: Small dataset for testing
dataset = [
    [1.1, 2.5, 0], [1.3, 2.3, 0], [1.5, 2.2, 0], [1.7, 2.8, 0],  # Class 0
    [2.1, 3.0, 1], [2.3, 3.2, 1], [2.5, 3.1, 1], [2.7, 3.6, 1]   # Class 1
]

# Split dataset into train and test data
train_data = dataset
test_data = [[1.4, 2.4], [2.6, 3.3]]  # Test without labels

# Run the Naive Bayes classifier on the test data
predictions = naive_bayes(train_data, test_data)

# Print predictions
for i in range(len(test_data)):
    print(f"Test instance {i + 1}: Predicted class = {predictions[i]}")