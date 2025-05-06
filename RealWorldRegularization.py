# Here’s a real-world example using the Breast Cancer dataset from Scikit-learn. We’ll apply L1 (Lasso) and L2 (Ridge) regularization to see how they affect feature selection and performance.

# Step 1: Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#Step 2: Load and Preprocess the Dataset
# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Split dataset into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for regularization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Step 3: Train Models with Regularization
# Train Logistic Regression models with L1 and L2 regularization
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)  # L1 (Lasso)
model_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=0.1)  # L2 (Ridge)

# Fit models to the training data
model_l1.fit(X_train, y_train)
model_l2.fit(X_train, y_train)

# Predict on test data
y_pred_l1 = model_l1.predict(X_test)
y_pred_l2 = model_l2.predict(X_test)

# Calculate accuracy
accuracy_l1 = accuracy_score(y_test, y_pred_l1)
accuracy_l2 = accuracy_score(y_test, y_pred_l2)
print(f"L1 Regularization Accuracy: {accuracy_l1:.4f}")
print(f"L2 Regularization Accuracy: {accuracy_l2:.4f}")

#Step 4: Compare Feature Weights (Effect of Regularization)
# Get feature coefficients
coef_l1 = model_l1.coef_.flatten()
coef_l2 = model_l2.coef_.flatten()

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_names, y=coef_l1, color="blue", alpha=0.6, label="L1 (Lasso)")
sns.barplot(x=feature_names, y=coef_l2, color="red", alpha=0.6, label="L2 (Ridge)")
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.title("Feature Importance with L1 and L2 Regularization")
plt.legend()
plt.show()

# Explanation of the Output
# L1 (Lasso) Regularization

# Some coefficients (feature weights) shrink to exactly zero → Feature selection happens automatically.
# Less relevant features are removed, simplifying the model.
# L2 (Ridge) Regularization

# Shrinks all weights but doesn’t set any to zero → Keeps all features but reduces their impact.
# Prevents over-reliance on specific features.
# Comparison of Accuracy

# Both models should have high accuracy but L1 may slightly outperform if irrelevant features exist.

# Key Takeaways
# ✅ L1 (Lasso) Regularization → Eliminates unimportant features (feature selection).
# ✅ L2 (Ridge) Regularization → Keeps all features but prevents overfitting.
# ✅ Regularization prevents overfitting and improves generalization.

# Interpreting the Coefficients in the Regularization Plot
# When you see bar plots of feature coefficients from L1 and L2 regularized Logistic Regression models, here’s how to interpret them:

# 1️⃣ Positive vs. Negative Coefficients
# Positive Coefficients (Above Zero)
# → These features increase the likelihood of the positive class (e.g., predicting "Malignant" in the breast cancer dataset).
# Negative Coefficients (Below Zero)
# → These features decrease the likelihood of the positive class (i.e., they support the opposite class, "Benign").
# Near Zero Coefficients
# → Features with values close to zero have little or no influence on the prediction.
# 💡 Example:
# If "tumor size" has a positive coefficient, a larger tumor size increases the chance of being classified as malignant.
# If "smoothness" has a negative coefficient, higher smoothness decreases the chance of being malignant.

# 2️⃣ L1 vs. L2 Effects on Coefficients
# L1 (Lasso) Regularization

# Some coefficients are exactly zero, meaning the model ignored those features as unimportant.
# Only a few important features remain with nonzero values.
# Useful for feature selection (identifying the most relevant variables).
# L2 (Ridge) Regularization

# No coefficients are exactly zero, but all are shrunk towards zero.
# All features are kept, but their influence is reduced.
# Prevents overfitting while maintaining all input variables.
# 3️⃣ Interpreting Feature Importance
# Tall Bars (High Absolute Value)

# These features strongly impact predictions.
# If a feature has a large positive or large negative coefficient, it plays a key role in classification.
# Short Bars (Near Zero)

# These features have little effect on predictions.
# In L1 regularization, some features disappear completely (zero weight), meaning they are not useful.
# 📌 Practical Interpretation Example (Breast Cancer Dataset)
#----------------------------------------------------------------------------------------------------
# Feature	   | L1 Coefficient	|L2 Coefficient	|Meaning
# Tumor Radius |	2.5	        |    1.8	    | A larger tumor increases chance of malignancy
# Smoothness   |    -1.2	    |    -0.8	    | Smoother tumors less likely to be malignant
# Compactness  |     0.0	    |     0.5	    | L1 removed this feature, L2 kept it but with low influence
#----------------------------------------------------------------------------------------------------
# 🔍 Key Takeaways
# ✅ Positive coefficients → Feature supports the positive class.
# ✅ Negative coefficients → Feature supports the negative class.
# ✅ L1 (Lasso) → Some coefficients become exactly zero (irrelevant features removed).
# ✅ L2 (Ridge) → All coefficients shrink but remain nonzero.
# ✅ Higher absolute values → More influence on prediction.

# Would you like to highlight the most important features in the plot automatically? 😊







