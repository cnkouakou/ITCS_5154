# ======================== Preprocessing and Data Splitting ========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset.
# Note: Change the file path as necessary.
path = "C:\\Users\\cnkou\\OneDrive\\Documents\\Education\\UNCC\ITCS 5154\\Module 4\\Code\\"
df = pd.read_csv(path + 'Maternal_Health_Risk_Data_Set.csv')
# Inspect the data briefly.
print("Dataset head:")
print(df.head())
"Head", df.head(10)
"df.shape", df.shape
"", df.columns
"Summary:", df.describe()
#------------------------------------------------------

#==========================================================
# -------------
# Data Preprocessing:
# -------------

# If the target is categorical (e.g., 'Low', 'Medium', 'High'), we convert it to numeric.
if df['RiskLevel'].dtype == 'object':
    le = LabelEncoder()
    df['RiskLevel'] = le.fit_transform(df['RiskLevel'])
    print("\nClasses found and encoded as:", list(le.classes_))

# Define features and target.
features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
X = df[features]
y = df['RiskLevel']

"Target", y
# Split the data into training and test sets.
X_train, X_test, t_train, t_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features (very important for many classifiers, especially those based on distance).
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"t_train", t_train


# ======================== Creating Classifiers ========================
# Import classifiers
from sklearn.linear_model import RidgeClassifier, Perceptron, SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Create instances for each classifier.
ridge       = RidgeClassifier()
perceptron  = Perceptron()
sgd         = SGDClassifier()
svm         = SVC()  
logreg      = LogisticRegression(max_iter=500)
knn         = KNeighborsClassifier()
nb          = GaussianNB()

# Lists of classifiers and names.
clfs = [ridge, perceptron, sgd, svm, logreg, knn, nb]
names = ["Ridge", "Perceptron", "SGD", "SVM", "LogReg", "kNN", "NaiveBayes"]

# ======================== Training, Evaluation, and Visualization ========================
train_accs = []
test_accs = []

for name, clf in zip(names, clfs):
    print("{:=^50s}".format(name))
    
    # Train the classifier.
    clf.fit(X_train, t_train)
    
    # Evaluate on training and test sets.
    train_score = clf.score(X_train, t_train)
    test_score = clf.score(X_test, t_test)
    
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    "Model:", name, ", Test Accuracy:", test_score, ", Train Accuracy:",  train_score
    
    # Save accuracies for the bar plot.
    train_accs.append(train_score)
    test_accs.append(test_score)
    
    # Make predictions.
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # Visualize predictions vs. true labels.
    # Create a figure with two subplots side-by-side
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot for training data predictions
    ax1.plot(t_train, 'ro', label='True Labels')
    ax1.plot(y_train_pred, 'bx', label='Predicted Labels')
    ax1.set_title(f"Train Predictions: {name}")
    ax1.legend()

    # Plot for test data predictions
    ax2.plot(t_test, 'ro', label='True Labels')
    ax2.plot(y_test_pred, 'bx', label='Predicted Labels')
    ax2.set_title(f"Test Predictions: {name}")
    ax2.legend()

    # Set the overall figure title
    fig1.suptitle(name)

    # Display the figure in Streamlit
    st.pyplot(fig1)

# ======================== Bar Plot Comparing Accuracies ========================
x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width/2, train_accs, width, label='Train Accuracy')
ax.bar(x + width/2, test_accs, width, label='Test Accuracy')
ax.set_xticks(x, names)
ax.set_xlabel('Classifier')
ax.set_ylabel('Accuracies')
ax.set_title('Comparison of Train and Test Accuracies on Maternal Health Risk Dataset')
ax.legend()
plt.show()
st.pyplot(fig)


#=======================================================
for name, clf in zip(names, clfs):
    print("{:=^50s}".format(name))
    
    # TODO 10.3
    clf.fit(X_train, t_train)
    
    # TODO 10.4
    train_score = clf.score(X_train, t_train)
    
    # TODO 10.5
    test_score = clf.score(X_test, t_test)
    
    print(f"Train Accuracy: {train_score}\nTest Accuracy: {test_score}")

    # TODO 10.6
    y_train = clf.predict(X_train)
    
    # TODO 10.7
    y_test = clf.predict(X_test)
    
    # Track each model/classifier's train and test accuracy
    train_accs.append(train_score)
    test_accs.append(test_score)
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(121)
    plt.plot(t_train, 'ro')
    plt.plot(y_train, 'bx')
    plt.title("Train")
    
    plt.subplot(122)
    plt.plot(t_test, 'ro')
    plt.plot(y_test, 'bx')
    plt.title("Test")
    plt.suptitle(name)
    plt.show()

    ====================
# TODO 10.8
x = np.arange(len(names))  
width = 0.35  
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, train_accs, width, label='Train Accuracy')
plt.bar(x + width/2, test_accs, width, label='Test Accuracy')
plt.xticks(x, names)
plt.xlabel('Classifier')
plt.ylabel('Accuracies')
plt.title('Comparison of Train and Test Accuracy for Classifiers')
plt.legend(loc='lower right')
plt.show()