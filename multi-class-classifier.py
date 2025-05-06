'''
Identifying classifiers

I'm working on selecting four classifiers that can inherently handle multi-class 
classification. Perceptron, SGD, Logistic Regression, kNN, and Naive Bayes are among the options, with SVM and RidgeClassifier also being considered.

Assessing multi-class capability

I’m mapping out which classifiers can inherently handle multi-class classification.
 RidgeClassifier, Perceptron, SGDClassifier, SVM (SVC), Logistic Regression, kNN, and Naive Bayes each offer unique multi-class features.

Unpacking the classifiers

OK, let me see. Perceptron, SGDClassifier, LogisticRegression, and 
RidgeClassifier natively support multi-class classification, either 
through one-vs-all or softmax, without needing a one-vs-rest wrapper.

Determining multi-class capabilities

I’m noting that SVM (SVC), kNN, Logistic Regression, and Naive Bayes 
inherently support multi-class classification. Based on documentation, 
RidgeClassifier does as well, while Perceptron and SGD might need 
further verification.

Determining multi-class classifiers

It seems that RidgeClassifier inherently handles multi-class classification, 
while Perceptron, SGD, and Logistic Regression may need a one-vs-rest wrapper, 
suggesting they are inherently binary classifiers.

Choosing the classifiers

I'm concluding that SVM (SVC), kNN, Logistic Regression, and Naive Bayes 
inherently support multi-class classification, while Perceptron, SGD, and 
RidgeClassifier are typically binary.

Highlighting key classifiers

In scikit-learn, SVM (SVC), Logistic Regression, kNN, and Naive Bayes are 
inherently multi-class, while Perceptron, SGD, and RidgeClassifier are typically 
binary.

Below is one complete solution that uses only the four classifiers which 
inherently support multi‐class classification (i.e. without wrapping them in 
OneVsRestClassifier). In this example, we use SVM, Logistic Regression, kNN, 
and Gaussian Naive Bayes. (Recall that SVC handles multi-class problems via a 
one‐vs‐one strategy internally.) You can run the code below to train, evaluate, 
and plot the predictions for each classifier.

'''
# TODO 9.1: Import necessary modules and create the classifiers that natively handle multi-class problems
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Import the classifiers
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Optional helper function for splitting data
def data_splitting(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Load the Iris dataset and split into training and test sets
iris = load_iris()
X_train, X_test, t_train, t_test = data_splitting(iris.data, iris.target)

# Create the instances of the classifiers (4 in total)
clfs = [SVC(), LogisticRegression(max_iter=200), KNeighborsClassifier(), GaussianNB()]

# TODO 9.2: Create the corresponding names list
names = ["SVM", "Logistic Regression", "kNN", "Naive Bayes"]

# Loop through each classifier
for name, clf in zip(names, clfs):
    print("{:=^50s}".format(name))
    
    # TODO 9.3: Train the classifier using X_train and t_train
    clf.fit(X_train, t_train)
    
    # TODO 9.4: Evaluate the model on the training data
    train_score = clf.score(X_train, t_train)
    
    # TODO 9.5: Evaluate the model on the test data
    test_score = clf.score(X_test, t_test)
    
    print(f"Train Accuracy: {train_score}\nTest Accuracy: {test_score}")
    
    # TODO 9.6: Make predictions for the training data and store in y_train
    y_train = clf.predict(X_train)
    
    # TODO 9.7: Make predictions for the test data and store in y_test
    y_test = clf.predict(X_test)
    
    # Plot the results for visualization
    plt.figure(figsize=(12,4))
    
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
