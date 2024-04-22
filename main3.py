# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:31:25 2021

@author: JINH
"""


#import pandas
import pandas as pd #data analisys and manipulation
import numpy as np  #Arrays, matrix, computation
import matplotlib.pyplot as plt
import seaborn as sns #statistical data visualization
import warnings
warnings.filterwarnings('ignore')

sns.set()
#%matplotlib inline

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 
             'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("diabetes.csv",  names=col_names)
pima.head()


#split dataset in features and target variable
feature_cols = ['pregnant', 'glucose','bp','skin','insulin', 'bmi', 
                'pedigree','age',]
X = pima[feature_cols] # Features
y = pima.label # Target variable

print(end="\n")
print(X)
print(end="\n")
print('etiquetas o salidas y')
print(y)

#Filtrar los datos, remover los ceros o sustituirlos
#cambiarlos por la desviacion estandar o la media
# let's see how data is distributed for every column.

plt.figure(figsize = (20, 25))
plotnumber = 1

for column in pima:
    if plotnumber <= 9:
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(pima[column]) #estadisticos
        plt.xlabel(column, fontsize = 15)
        
    plotnumber += 1
plt.show()

#estadisticas de las columnas
print(end="\n")
print("Estadisticas de las columnas")
Estadisticas=pima.describe()
print(Estadisticas)
print(end="\n")
#graficar y estadisticas para datos filtrados


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#print(X_train)
#print(X_test)

# Modulo de logisctic regression
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

#Para ver los coeficientes encontrados.
trained_regressor = LogisticRegression().fit(X, y)
print(end="\n")
print("coeficientes")
coe=trained_regressor.coef_
print(coe)
print(end="\n")
#Para ver b0 o bias

bias=trained_regressor.intercept_
print('bias:', bias)
print(end="\n")

#Para ver el modelo de regresion o de decision completo
#from LinearRegressor.decison_function
#scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_


# import the metrics class, para evaluar la matriz de confusion
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
print("Matrix de confusion")
print(cnf_matrix)
print(end="\n")
#Visualizing Confusion Matrix using Heatmap
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#Text(0.5,257.44,'Predicted label')

#Confusion Matrix Evaluation Metrics
print('Accuracy:',metrics.accuracy_score(y_test, y_pred))
print(end="\n")
print("Precision:",metrics.precision_score(y_test, y_pred))
print(end="\n")
print("Recall:",metrics.recall_score(y_test, y_pred))
