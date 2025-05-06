# TODO: Import necessary modules
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Import classifiers
from sklearn.linear_model import RidgeClassifier, Perceptron, SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Import OneVsRestClassifier for wrapping binary classifiers in a multiclass setup
from sklearn.multiclass import OneVsRestClassifier

# Optionally, define a data_splitting helper function if needed
def data_splitting(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Load the iris dataset
iris = load_iris()
X_train, X_test, t_train, t_test = data_splitting(iris.data, iris.target)

# TODO 8.1: Create instances of each classifier as described
ridge = RidgeClassifier()
perceptron = Perceptron()
sgd = SGDClassifier()
svm = SVC()  # SVM classifier (default kernel)
logreg = LogisticRegression(max_iter=200)  # Increase max_iter if needed
knn = KNeighborsClassifier()
nb = GaussianNB()

# list of algorithms to test
clfs = [ridge, perceptron, sgd, svm, logreg, knn, nb]
# list of algorithm names 
names = ["Ridge", "Perceptron", "SGD", "SVM", "Logistic Regression", "kNN", "Naive Bayes"]

# Loop through classifiers defined above
for name, base_clf in zip(names, clfs):
    print("{:=^50s}".format(name))
    
    # TODO 8.2: Wrap the current classifier in OneVsRestClassifier
    clf = OneVsRestClassifier(base_clf)
    
    # TODO 8.3: Train the OneVsRestClassifier instance using X_train and t_train
    clf.fit(X_train, t_train)
    
    # TODO 8.4: Evaluate the model by computing the training score
    train_score = clf.score(X_train, t_train)
    
    # TODO 8.5: Evaluate the model by computing the test score
    test_score = clf.score(X_test, t_test)
    
    print(f"Train Accuracy: {train_score}\nTest Accuracy: {test_score}")
    
    # TODO 8.6: Make predictions for the train data and store in y_train
    y_train = clf.predict(X_train)
    
    # TODO 8.7: Make predictions for the test data and store in y_test
    y_test = clf.predict(X_test)
    
    # Plotting the results for visualization
    plt.figure(figsize=(12, 4))
    
    # Train predictions plot
    plt.subplot(121)
    plt.plot(t_train, 'ro', label='True Labels')
    plt.plot(y_train, 'bx', label='Predicted Labels')
    plt.title("Train")
    plt.legend()
    
    # Test predictions plot
    plt.subplot(122)
    plt.plot(t_test, 'ro', label='True Labels')
    plt.plot(y_test, 'bx', label='Predicted Labels')
    plt.title("Test")
    plt.legend()
    
    plt.suptitle(name)
    plt.show()
