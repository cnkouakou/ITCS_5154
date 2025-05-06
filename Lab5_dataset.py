import pandas as pd
from ucimlrepo import fetch_ucirepo 
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
print(X) 
print(y) 
# metadata 
print(wine_quality.metadata) 

df_wine = pd.concat([X, y], axis = 1)

# Plot: Correlation Heatmap
# A heatmap helps visualize relationships between features.

plt.figure(figsize=(10, 8))
sns.heatmap(df_wine.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#  Observations:
# Alcohol has a strong positive correlation with quality.
# Volatile acidity has a strong negative correlation with quality.
# Some features (like density and fixed acidity) are weakly correlated with quality.


#======================================================================

# Histogram of wine qualite:
# Plot wine quality distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="quality", data=df_wine, palette="viridis")
plt.title("Distribution of Wine Quality Ratings")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

# 🔍 Observations:
# Most wines are rated between 5 and 6, with very few extreme ratings (3 or 8).
# The dataset is imbalanced, which may impact machine learning models.

scatter_matrix(df_wine,
               figsize=(15, 15),
               diagonal='kde',  
               c=df_wine['quality'],  
               cmap='viridis',  
               alpha=0.3)
plt.show()

# Create a pairplot
sns.pairplot(df_wine, markers= '.', diag_kind='hist')  # 'diag_kind' sets histogram for diagonal plots

# Show the plot
plt.show()

# variable information 
print(wine_quality.variables) 


#==================================================================
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
energy_efficiency = fetch_ucirepo(id=242) 
  
# data (as pandas dataframes) 
X = energy_efficiency.data.features 
y = energy_efficiency.data.targets 
print(X.columns)
df_energy = pd.concat([X, y], axis = 1)

plt.figure(figsize=(10, 8))
sns.heatmap(df_energy.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#print(df_energy)
sns.pairplot(df_wine, markers= '.', diag_kind='hist')  # 'diag_kind' sets histogram for diagonal plots
# metadata 
print(energy_efficiency.metadata) 
  
# variable information 
print(energy_efficiency.variables)