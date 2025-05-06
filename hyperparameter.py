# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Step 1: Define the hyper-parameter grid
param_grid = {
    'sgd__alpha': [0.0001, 0.001, 0.01, 0.1],  # Regularization strength
    'sgd__penalty': ['l2', 'l1', 'elasticnet'],  # Type of regularization
    'sgd__loss': ['squared_error', 'huber', 'epsilon_insensitive'],  # Loss function
    'sgd__max_iter': [1000, 2000, 3000]  # Max iterations
}

# Step 2: Create a pipeline (StandardScaler + SGDRegressor)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDRegressor(random_state=0))
])

# Step 3: Create GridSearchCV instance
gscv = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)

# Step 4: Prepare the dataset
# Combine targets (Y1, Y2) into a single target column
df_energy['combined_target'] = df_energy[['Y1', 'Y2']].mean(axis=1)

# Step 5: Fit the model
gscv.fit(df_energy_features, df_energy['combined_target'])

# Step 6: Get the best hyper-parameters
best_params = gscv.best_params_

# Print the best parameters
print("Best hyperparameters:", best_params)


from sklearn.model_selection import train_test_split

# Assuming df_wine is already loaded and contains the "quality" column
X = df_wine.drop(columns=['quality'])  # Features
t = df_wine['quality']  # Target

# Splitting data (80% training, 20% testing)
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=0)

from sklearn.feature_selection import VarianceThreshold

# Initialize VarianceThreshold with threshold 0.1
vt = VarianceThreshold(threshold=0.1)


vt_X_train = vt.fit_transform(X_train)

# Get feature mask (True = Selected, False = Removed)
selected_mask = vt.get_support()

# Get names of selected features
selected_features = X_train.columns[selected_mask]

print(f"Number of selected features: {len(selected_features)}")
print("Selected features:", list(selected_features))
