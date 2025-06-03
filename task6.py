import csv
import math
from collections import Counter

# Load iris dataset from CSV
def load_iris_dataset(filename):
    data = []
    labels = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                features = list(map(float, row[:4]))
                label = row[4].strip()
                data.append(features)
                labels.append(label)
            except ValueError:
                continue  # Skip rows with invalid data
    return list(zip(data, labels))

# Normalize dataset using Min-Max scaling
def normalize_dataset(dataset):
    features = [item[0] for item in dataset]
    min_vals = [min(col) for col in zip(*features)]
    max_vals = [max(col) for col in zip(*features)]

    normalized = []
    for item in dataset:
        normalized_features = [
            (val - minv) / (maxv - minv) if maxv != minv else 0.0
            for val, minv, maxv in zip(item[0], min_vals, max_vals)
        ]
        normalized.append((normalized_features, item[1]))
    return normalized

# Split into train and test sets (80% train, 20% test)
def train_test_split(dataset, test_ratio=0.2):
    split_idx = int(len(dataset) * (1 - test_ratio))
    return dataset[:split_idx], dataset[split_idx:]

# Euclidean distance
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# k-NN prediction
def knn_predict(train_data, test_instance, k=3):
    distances = [(euclidean_distance(test_instance, train[0]), train[1]) for train in train_data]
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    labels = [label for _, label in k_nearest]
    return Counter(labels).most_common(1)[0][0]

# Accuracy calculation
def calculate_accuracy(true_labels, predicted_labels):
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    return correct / len(true_labels) * 100

# Confusion matrix generation
def confusion_matrix(true_labels, predicted_labels, class_names):
    matrix = [[0 for _ in class_names] for _ in class_names]
    for true, pred in zip(true_labels, predicted_labels):
        if true not in class_names or pred not in class_names:
            continue
        i = class_names.index(true)
        j = class_names.index(pred)
        matrix[i][j] += 1
    return matrix

# Print confusion matrix
def print_confusion_matrix(cm, class_names):
    print("\nConfusion Matrix:")
    print(f"{'':15s}" + "".join(f"{name:15s}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:15s}" + "".join(f"{val:<15d}" for val in row))

# ==== MAIN EXECUTION ====
if __name__ == "__main__":
    dataset = load_iris_dataset("iris.csv")
    dataset = normalize_dataset(dataset)
    train_data, test_data = train_test_split(dataset)

    predictions = []
    true_labels = []

    for features, label in test_data:
        predicted = knn_predict(train_data, features, k=3)
        predictions.append(predicted)
        true_labels.append(label)

    accuracy = calculate_accuracy(true_labels, predictions)
    print(f"\nAccuracy: {accuracy:.2f}%")

    all_classes = sorted(set(true_labels + predictions))
    cm = confusion_matrix(true_labels, predictions, all_classes)
    print_confusion_matrix(cm, all_classes)
