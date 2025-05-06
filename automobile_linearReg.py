
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
# TODO: Import all the linear regression models that you used here
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# LOADING DATA
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/10/data.csv"
)

"""
# 1985 Auto Imports Database

Abstract: This data set consists of three types of entities:
        (a) the specification of an auto in terms of various characteristics,
        (b) its assigned insurance risk rating,
        (c) its normalized losses in use as compared to other cars. The second rating
        corresponds to the degree to which the auto is more risky than its price indicates.
        Cars are initially assigned a risk factor symbol associated with its
        price.   Then, if it is more risky (or less), this symbol is
        adjusted by moving it up (or down) the scale.  Actuarians call this
        process "symboling".  A value of +3 indicates that the auto is
        risky, -3 that it is probably pretty safe.
"""

@st.cache_data
def load_data(nrows):
    # TODO: Import the data, preprocess it, and seperate it into training and testing data
    df = pd.read_csv(DATA_URL, nrows=nrows)
    #is there missing values
    null_df = df.isnull()
    null_df

    rows_with_null = df[df.isnull().any(axis=1)]
    rows_with_null
    #Using SimpleImputer in Scikit-Learn, replace the missing values (NaN) with the most 
    #frequent values in the data. Store the cleaned data into df_freq.
    imputer = SimpleImputer(strategy='most_frequent')
    df_freq = pd.DataFrame(imputer.fit_transform(df))
    df_freq
    df_freq.head()
    df_freq.columns = df.columns
    df_freq.head()

    df = df_freq
    "Dataframe", df.head()
    
    #Let's check the type of data in each column
    df.dtypes
    numeric_columns = ['normalized-losses', 'num-of-doors', 'wheel-base', 'length', 'width', 'height',
                    'curb-weight', 'num-of-cylinders', 'engine-size', 'bore', 'stroke', 'compression-ratio',
                    'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price', 'symboling']

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    df.dtypes
    #Convert the data present in the form of strings in df to integer format using labelEncoder function. 
    #The list of columns with string data is given to you in the form of a list stored in strings_list
    strings_list = ['make', 'fuel-type', 'aspiration', 'body-style',
                'drive-wheels', 'engine-location', 'engine-type', 'fuel-system']

    label_encoder = preprocessing.LabelEncoder()
    for col in strings_list:
        df[col] = label_encoder.fit_transform(df[col])

    df.dtypes
    # let's lookk at df
    "DataFrame df now: ", df


    T = df.loc[:, 'symboling'].copy() # copy the target into T
    X = df.iloc[:, :-1].copy()         # copy the feature into X
    N = df.shape[0]
    "X.shape", X.shape

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, t_train, t_test = train_test_split(X, T, test_size=0.2, random_state=0)
    X.shape, X_train.shape, X_test.shape, t_train.shape, t_test.shape

    return df, X_train, X_test, t_train, t_test
    
df, X_train, X_test, t_train, t_test = load_data(100000)


"## Summary"
st.dataframe(df.describe())


#################### functions

def evaluate(y, t):
    fig = plt.figure(figsize=(10,10))

    # Paste the corresponding part of your evaluate() function
    # t vs y plot
    plt.subplot(3,3, 1)
    # TODO: add the first plot
    plt.scatter(t_test, y, alpha=0.7, marker = '.')
    # dashed diagonal line
    plt.plot([-3,3], [-3,3], 'r--')
    plt.xlabel("target")
    plt.ylabel("Predicted")
    plt.title("Plot-1")

    # all value comparison
    plt.subplot(3,2, 2)
    # TODO: add the second one
    plt.plot(t_test.to_numpy(), '.')
    plt.plot(y, 'x')
    plt.xlabel("samples")
    plt.ylabel("symboling")
    plt.title("Plot-2")
    st.pyplot(fig)  

    # subplots of individual quality comparision
    # TODO: add the third subplots
    unique_values = sorted(dict.fromkeys(t_test.values)) 
    fig , axes = plt.subplots(2, 3, figsize=(10,6)) 
    axes = axes.ravel()  # Flatten the axes for easier iteration

    for i, val in enumerate(unique_values):
        axes[i].plot(t_test.values[t_test==val], '.', label="Actual")
        axes[i].plot(y[t_test==val], 'x', label="Predicted")
        axes[i].set_ylabel(f"Actual Symboling = {val}")
        axes[i].grid(alpha=0.3)
    # Hide any unused subplots
    for j in range(len(unique_values), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    st.pyplot(fig)

# print the value text over the bar
# https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
def autolabel(ax, rects):
    # Paste the corresponding part of your autolabel() function
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:0.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color='blue')
            

def show_weights(model, names):

    # combine both the coefficients and intercept to present
    w = np.append(model.coef_, model.intercept_)

    fig = plt.figure(figsize=(12,3))

    # Paste the corresponding part of your show_weights() function
    names = list(names) + ['bias/intercept']

    # create bar chart to present the weights
    rects = plt.bar(range(len(w)), w, color='skyblue')

    ax = plt.gca()
    ax.set_xticks(range(len(w)))
    ax.set_xticklabels(names, rotation = 90)

    # TODO: call the autolabel function
    autolabel(ax, rects)
    plt.title("Model Weights and Bias")
    plt.ylabel("Value")
    #plt.tight_layout()

    st.pyplot(fig)
####################


st.divider()
# TODO: Add your code to observe different models
'''
# Linear Regression
'''
np.random.seed(0)

# 1) initialize
# create a linear regression object
model = LinearRegression()
# 2) train the model
model.fit(X_train, t_train)
# 3) evaluate
test_score = model.score(X_test, t_test)
test_score
# 4) Predict the values
y = model.predict(X_test)
# 5) Plot actual vs predicted graphs
evaluate(y, t_test.to_numpy())
# 6) Plot the weights
show_weights(model, df.columns.values[:-1])

st.divider()

'''
# Ridge Regression
Now let's repeat everything we just did but now let's look at using linear regression with regularization. 
To start off, let's try using ridge regression and see if the result differ at all!
'''
# 1) initialize
model = Ridge(alpha=1.0) 

# 2) train the model
model.fit(X_train, t_train)

# 3) evaluate
test_score = model.score(X_test, t_test)
"Test score: ", test_score

# 4) Predict the values
y = model.predict(X_test)
# 5) Plot actual vs predicted graphs
evaluate(y, t_test.to_numpy())
# 6) Plot the weights
show_weights(model, df.columns.values[:-1])

st.divider()

'''
# Lasso Regression
'''
# 1) initialize
model = Lasso(alpha=0.1)
# 2) train the model
model.fit(X_train, t_train)
# 3) evaluate
test_score = model.score(X_test, t_test)
"Test score: ", test_score

# 4) Predict the values
y = model.predict(X_test)
# 5) Plot actual vs predicted graphs
evaluate(y, t_test.to_numpy())
# 6) Plot the weights
show_weights(model, df.columns.values[:-1])

st.divider()

'''
# Elastic Net
'''
# 1) initialize
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
# 2) train the model
model.fit(X_train, t_train)
# 3) evaluate
test_score = model.score(X_test, t_test)
"Test score: ", test_score

# 4) Predict the values
y = model.predict(X_test)
# 5) Plot actual vs predicted graphs
evaluate(y, t_test.to_numpy())
# 6) Plot the weights
show_weights(model, df.columns.values[:-1])
st.divider()

'''
# Stochastic Gradient Descent
'''
# 1) initialize
model = SGDRegressor(max_iter=1000, tol=1e-3, penalty='l2', alpha=0.03, random_state=42)

# 2) train the model
model.fit(X_train, t_train)
# 3) evaluate
test_score = model.score(X_test, t_test)
"Test score: ", test_score

# 4) Predict the values
y = model.predict(X_test)
# 5) Plot actual vs predicted graphs
evaluate(y, t_test.to_numpy())
# 6) Plot the weights
show_weights(model, df.columns.values[:-1])

st.divider()

