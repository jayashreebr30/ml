# ml
# 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

# Load the dataset
data = pd.read_csv("iris_1.csv")

# Display first 5 rows
print("First 5 rows of dataset:")
print(data[:5])

# Extract features and labels
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

print("\nShape of Features and Labels:")
print(X.shape, y.shape)

# Standardizing the features using Z-score normalization
X = StandardScaler().fit_transform(X)

print("\nStandardized values:")
print(X[:5])

# Initializing PCA with 2 principal components
pca1 = decomposition.PCA(n_components=2)

print("\nPCA Object:")
print(pca1)

# Transforming data using PCA
pca_data = pca1.fit_transform(X)

# Eigen values (Explained Variance)
print("\nExplained Variance (Eigen values) of each Principal Component:")
print(pca1.explained_variance_)

# Variance ratio
percentage_var = pca1.explained_variance_ / np.sum(pca1.explained_variance_)

# Cumulative variance
cumulative_var = np.cumsum(percentage_var)

print("\nExplained Variance Ratio of each Principal Component:")
print(percentage_var)

print("\nCumulative Variance Ratio:")
print(cumulative_var)

print("\nShape of PCA transformed data:")
print(pca_data.shape)

print("\nFirst 5 Transformed Data Points:")
print(pca_data[:5])

# Convert PCA output to DataFrame
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])

# Add target labels
pca_df['variety'] = y

# Define colors for each species
target_colors = {
    'Setosa': 'red',
    'Versicolor': 'blue',
    'Virginica': 'green'
}

# Plot PCA results
plt.figure(figsize=(8, 6))

for species, color1 in target_colors.items():
    subset = pca_df[pca_df['variety'] == species]

    plt.scatter(
        subset['PC1'],
        subset['PC2'],
        label=species,
        color=color1
    )

# Compute Eigenvectors
eigenvectors = pca1.components_

# Mean point of transformed data
mean_x, mean_y = np.mean(pca_data, axis=0)

pc_colors = ['red', 'black']

# Plot PCA component axes
for i in range(2):
    plt.arrow(
        mean_x,
        mean_y,
        eigenvectors[i, 0],
        eigenvectors[i, 1],
        color=pc_colors[i],
        head_width=0.1,
        label=f'PC{i+1}'
    )

# Customize plot
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset with Component Axes')

plt.legend()
plt.grid(True)

plt.show()









# 2


#importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
#importing datasets
data_set= pd.read_csv('Position_Salaries.csv')
#Extracting Independent and dependent Variable
x= data_set.iloc[:, 1:2].values
y= data_set.iloc[:, 2].values
print('position level and the salaries\n',x,y)

#Fitting the Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regs= LinearRegression()
lin_regs.fit(x,y)

#Visulaizing the result for Linear Regression model
mtp.scatter(x,y,color="blue")
mtp.plot(x,lin_regs.predict(x), color="red")
mtp.title('Bluff detection model(Linear Regression)')
mtp.xlabel('Position Levels')
mtp.ylabel('Salary')
mtp.show()

#Fitting the Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regs= PolynomialFeatures(degree= 3)
x_poly= poly_regs.fit_transform(x)
print('Polynomial coefficents for given degree are\n', x_poly)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly, y)

mtp.scatter(x,y,color='blue')
mtp.plot(x,lin_reg_2.predict(x_poly), color='red')
mtp.title('Bluff detection model(Polynomial Regression)')
mtp.xlabel('Position Levels')
mtp.ylabel('Salary')
mtp.show()

lin_pred = lin_regs.predict([[11]])  
print('prediction value using Linear Regression for the given level is\n',lin_pred)

poly_pred = lin_reg_2.predict(poly_regs.fit_transform([[11]]))  
print('prediction value using Polynomial Regression for the given level is\n',poly_pred)







# 3
import numpy as np
import pandas as pd

# Column names
col_names = [
    'pregnant', 'glucose', 'bp', 'skin',
    'insulin', 'bmi', 'pedigree', 'age', 'label'
]

# Load dataset
pima = pd.read_csv(
    "pima-indians-diabetes.csv",
    names=col_names
)

print(pima)

# Split dataset into features and target variable
feature_cols = [
    'pregnant', 'glucose', 'bp', 'skin',
    'insulin', 'bmi', 'pedigree', 'age'
]

# Features
x = pima[feature_cols]

# Target variable
y = pima.label

print(x, y)

# Split x and y into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=2
)

print("x values for training\n", x_train[0:5])
print("x values for testing\n", x_test[0:5])

print("y values for training\n", y_train[0:5])
print("y values for testing\n", y_test[0:5])

# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Initialize model
logreg = LogisticRegression()

# Fit the model
logreg.fit(x_train, y_train)

# Predict values
y_pred = logreg.predict(x_test)

print("Predicted values for the test data\n", y_pred)

# Import metrics
from sklearn import metrics

print("Actual Vs Predicted values")
print(np.column_stack((y_test, y_pred)))

# Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print("Confusion Matrix is")
print(cnf_matrix)

# Accuracy score
from sklearn.metrics import accuracy_score

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred, normalize=True) * 100
)





# 4

import pandas as pd
import numpy as np

# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Import scikit-learn metrics module
from sklearn import metrics

# Column names
col_names = [
    'pregnant', 'glucose', 'bp', 'skin',
    'insulin', 'bmi', 'pedigree', 'age', 'label'
]

# Load dataset
pima = pd.read_csv(
    "pima-indians-diabetes.csv",
    header=None,
    names=col_names
)

# Split dataset into features and target variable
feature_cols = [
    'pregnant', 'glucose', 'bp', 'skin',
    'insulin', 'bmi', 'pedigree', 'age'
]

# Features
x = pima[feature_cols]

# Target variable
y = pima.label

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=4
)

print(x_train, x_test)
print(y_train, y_test)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier(criterion='entropy')

# Train Decision Tree Classifier
clf = clf.fit(x_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy
print(
    "Accuracy:",
    metrics.accuracy_score(y_test, y_pred)
)

# Actual vs Predicted
print(np.column_stack((y_test, y_pred)))

# Export Decision Tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()

export_graphviz(
    clf,
    out_file=dot_data,
    filled=True,
    rounded=True,
    special_characters=True,
    feature_names=feature_cols,
    class_names=['0', '1']
)

graph = pydotplus.graph_from_dot_data(
    dot_data.getvalue()
)

graph.write_png('diabetes.png')

Image(graph.create_png())


# 5
# importing required libraries
# importing Scikit-learn library and datasets package
from sklearn import datasets
import pandas as pd
 # Loading the iris plants dataset (classification)
iris = datasets.load_iris()  
print(iris.target_names)
X, y = datasets.load_iris( return_X_y = True)
# Spliting arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
from sklearn.ensemble import RandomForestClassifier
# creating dataframe of IRIS dataset
data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],
                     'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],
                     'species': iris.target})
print(data.head())
# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
 # metrics are used to find accuracy or error
from sklearn import metrics
print()
 
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
# predicting which type of flower it is.
prediction = clf.predict([[3,3,2,2]])
print("Prediction for new instance:", iris.target_names[prediction][0])


# 6
Develop a program to
construct Support Vector Machine algorithm considering a Sample Dataset.
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take
the first two features. We could
 # avoid this ugly slicing by using a two-dim
dataset
y = iris.target
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear',
C=1,gamma=0.1).fit(X, y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:,
0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:,
1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min,
x_max, h),
 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(),
yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,
cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y,
cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
