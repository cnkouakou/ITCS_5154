# TODO 10.1: Import necessary modules and split the cancer data

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer

# Helper function for splitting data (if not already defined)
from sklearn.model_selection import train_test_split
def data_splitting(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Load the breast cancer dataset
cancer = load_breast_cancer()
X_train, X_test, t_train, t_test = data_splitting(cancer.data, cancer.target)

# TODO 10.2: Import classifier modules and create instances of the classifiers

from sklearn.linear_model import RidgeClassifier, Perceptron, SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Create classifier instances as specified:
ridge = RidgeClassifier()
perceptron = Perceptron()
sgd = SGDClassifier()
svm = SVC()  # Note: SVC supports multiclass classification via a one-vs-one approach.
logreg = LogisticRegression(max_iter=200)  # Increased iterations for convergence
knn = KNeighborsClassifier()
nb = GaussianNB()

# Define the lists of classifiers and their names
clfs = [ridge, perceptron, sgd, svm, logreg, knn, nb]
names = ["Ridge", "Perceptron", "SGD", "SVM", "LogReg", "kNN", "NaiveBayes"]

# Prepare lists to track the train and test accuracies for all classifiers
train_accs = []
test_accs = []

# Loop through each classifier
for name, clf in zip(names, clfs):
    print("{:=^50s}".format(name))
    
    # TODO 10.3: Train the classifier instance using X_train and t_train
    clf.fit(X_train, t_train)
    
    # TODO 10.4: Evaluate the model on the training data
    train_score = clf.score(X_train, t_train)
    
    # TODO 10.5: Evaluate the model on the test data
    test_score = clf.score(X_test, t_test)
    
    print(f"Train Accuracy: {train_score}\nTest Accuracy: {test_score}")
    
    # TODO 10.6: Make predictions for the train data and store in y_train_pred
    y_train_pred = clf.predict(X_train)
    
    # TODO 10.7: Make predictions for the test data and store in y_test_pred
    y_test_pred = clf.predict(X_test)
    
    # Save the accuracies for the bar plot later
    train_accs.append(train_score)
    test_accs.append(test_score)
    
    # Plotting true vs predicted labels for both train and test sets
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.plot(t_train, 'ro', label='True Labels')
    plt.plot(y_train_pred, 'bx', label='Predicted Labels')
    plt.title("Train")
    plt.legend()
    
    plt.subplot(122)
    plt.plot(t_test, 'ro', label='True Labels')
    plt.plot(y_test_pred, 'bx', label='Predicted Labels')
    plt.title("Test")
    plt.legend()
    
    plt.suptitle(name)
    plt.show()
   

# TODO 10.8: Create a bar plot that compares the train and test accuracies of all the algorithms

x = np.arange(len(names))  # positions for each classifier
width = 0.35                # width of each bar

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, train_accs, width, label='Train Accuracy')
plt.bar(x + width/2, test_accs, width, label='Test Accuracy')
plt.xticks(x, names)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Comparison of Train and Test Accuracy for Classifiers')
plt.legend()
plt.show()
