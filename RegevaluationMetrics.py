# Add necessary import statements
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# TODO 11.1
models = [LinearRegression, Ridge, Lasso, ElasticNet]

# Does not have any hyper-parameters (at least closed form doesn't)
linear_reg_params = {}

# TODO 11.2
ridge_params = {
'alpha': 1000
}

# TODO 11.3
lasso_params = {
 'alpha': 10   
}

# TODO 11.4
elastic_params = {
   'alpha': 0.1, 'l1_ratio': 0.7 
}

names = [ 'LinearReg', 'Ridge', 'Lasso', 'Elastic']
model_params = [linear_reg_params, ridge_params, lasso_params, elastic_params]

eval_metric_names = [
    "R2",
    'max_err',
    "MAE",
    "MAPE",
    "MSE"
]

# iterate through each target
for col in df_energy_targets.columns:
    
    print("Predicting ", col)
    df_results = []
    for name, model, params in zip(names, models, model_params):
        model = Pipeline([('scaler', StandardScaler()),
                           ('regr', model(**params))])  
        # TODO 11.5 - 11.6
        # Train model
        model.fit(X_train, t_train[col])
        # Predict on test set
        y_test = model.predict(X_test)
        ##### evaluate
         # Compute evaluation metrics
        r2 = r2_score(t_test[col], y_test)
        max_err_val = max_error(t_test[col], y_test)
        mae = mean_absolute_error(t_test[col], y_test)
        mape = mean_absolute_percentage_error(t_test[col], y_test)
        mse = mean_squared_error(t_test[col], y_test)  
        
        # TODO 11.7
        results = [
            r2, 
            max_err_val,
            mae,
            mape,
            mse  
        ]
        df_results.append(results)
    
    # Create metric dataframe for displaying results
    df_results = pd.DataFrame(
        df_results,
        index=names,
        columns=eval_metric_names)
    display(df_results)
    