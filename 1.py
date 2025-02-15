import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns

# Load the dataset from a CSV file
disease_df = pd.read_csv("framingham.csv")

# Remove the 'education' column from the dataset
disease_df.drop(['education'], inplace = True, axis = 1)

# Rename the 'male' column to 'Sex_male'
disease_df.rename(columns ={'male':'Sex_male'}, inplace = True)

# Remove rows with missing values
disease_df.dropna(axis = 0, inplace = True)

# Print the first few rows of the dataset and its shape (number of rows and columns)
print(disease_df.head(), disease_df.shape)

# Print the count of each value in the 'TenYearCHD' column
print(disease_df.TenYearCHD.value_counts())

# Select specific columns for features (X) and target (y)
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 
                           'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

# Normalize the feature data to have a mean of 0 and standard deviation of 1
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the data into training and testing sets (70% training, 30% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size = 0.3, random_state = 4)

# Print the shape of the training and testing sets
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Plot the count of patients affected with CHD (Coronary Heart Disease)
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df,
             palette="BuGn_r")
plt.show()

# Create a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# Train the model using the training data
logreg.fit(X_train, y_train)

# Predict the target values for the testing data
y_pred = logreg.predict(X_test)

# Calculate and print the accuracy of the model
from sklearn.metrics import accuracy_score
print('Accuracy of the model is =', 
      accuracy_score(y_test, y_pred))

# Create a confusion matrix to evaluate the performance of the model
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame for the confusion matrix with labeled columns and rows
conf_matrix = pd.DataFrame(data = cm, 
                           columns = ['Predicted:0', 'Predicted:1'], 
                           index =['Actual:0', 'Actual:1'])

# Plot the confusion matrix as a heatmap
plt.figure(figsize = (8, 5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens")

plt.show()

# Print the classification report for detailed evaluation metrics
print('The details for confusion matrix is =')
print (classification_report(y_test, y_pred))

