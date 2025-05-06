import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# 📌 Define models as references (DO NOT initialize yet)
models = [LinearRegression, Ridge, Lasso, ElasticNet]
names = ["Linear", "Ridge", "Lasso", "ElasticNet"]

# 📌 Set hyperparameters for regularized models
ridge_params = {'alpha': 100}  # Arbitrary, can be optimized later
lasso_params = {'alpha': 10}   # Arbitrary, can be optimized later
elastic_params = {'alpha': 0.1, 'l1_ratio': 0.7}  # Arbitrary, can be optimized later

# 📌 Define evaluation metrics
eval_metric_names = ["R2", "max_err", "MAE", "MAPE", "MSE"]

# 📌 Store results
results_df = pd.DataFrame(columns=["Target", "Model"] + eval_metric_names)

# 📌 Split features & targets
X_train, X_test, t_train, t_test = train_test_split(
    df_energy_features, df_energy_targets, test_size=0.2, random_state=0
)

# 📌 Outer loop: Iterate over targets (Y1, Y2)
for col in t_train.columns:
    # 📌 Inner loop: Iterate over models
    for model, name in zip(models, names):
        
        # Initialize model with parameters
        if name == "Ridge":
            reg = model(**ridge_params)
        elif name == "Lasso":
            reg = model(**lasso_params)
        elif name == "ElasticNet":
            reg = model(**elastic_params)
        else:
            reg = model()
        
        # Train model
        reg.fit(X_train, t_train[col])
        
        # Predict on test set
        y_test = reg.predict(X_test)

        # Compute evaluation metrics
        r2 = r2_score(t_test[col], y_test)
        max_err_val = max_error(t_test[col], y_test)
        mae = mean_absolute_error(t_test[col], y_test)
        mape = mean_absolute_percentage_error(t_test[col], y_test)
        mse = mean_squared_error(t_test[col], y_test)

        # Append results
        results_df = results_df.append(
            {
                "Target": col,
                "Model": name,
                "R2": r2,
                "max_err": max_err_val,
                "MAE": mae,
                "MAPE": mape,
                "MSE": mse
            },
            ignore_index=True
        )

# 📌 Print final results
print(results_df)
