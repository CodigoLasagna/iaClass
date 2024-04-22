import matplotlib.pyplot as plt
import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler# para normalizar

import seaborn as sns
from sklearn import metrics

#Carga los datos, vienen con sklearn
from sklearn.datasets import load_digits
#carga los datos como objeto
digits = load_digits()
digits

#
# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print('Image Data Shape', digits.data.shape)
# Print to show there are 1797 labels (integers from 0–9)
print("Label Data Shape", digits.target.shape)
#
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:10], digits.target[0:10])):
 plt.subplot(1, 11, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)
 
 
 #carga los datos como matriz
 x, y = load_digits(return_X_y=True)
 
 #Datos de entranamiento y prueba
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

#print(len(x_test)) #longitud de x_test
#print(len(y_test)) #longitud de y_test

#contenido de x_test[449]
imagen=x_test[449]
print(x_test[449])
print(imagen)

#Convierte la matriz en una imagen
plt.imshow(np.reshape(imagen, (8,8)), cmap=plt.cm.gray)


#Step 1. Import the model you want to use
from sklearn.linear_model import LogisticRegression

#Step 2. Make an instance of the Model
# all parameters not specified are set to their defaults
#LogisticRegression(C=0.05, class_weight=None, dual=False, fit_intercept=True,
#                   intercept_scaling=1, l1_ratio=None, max_iter=100,
#                   multi_class='ovr', n_jobs=None, penalty='l2', random_state=0,
#                   solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
logisticRegr = LogisticRegression()

#Step 3. Training the model on the data, storing the information learned from the data
#Model is learning the relationship between digits (x_train) and labels (y_train)
logisticRegr.fit(x_train, y_train)

#Step 4. Predict labels for new data (new images)
#Uses the information the model learned during the model training process
# Returns a NumPy Array
# Predict for One Observation (image1)
imagen1=x_test[0]
imagen1
plt.imshow(np.reshape(imagen1, (8,8)), cmap=plt.cm.gray)

imagen1a=logisticRegr.predict(x_test[0].reshape(1,-1))

#ver el resultado de la prediccion
#el vector "y" son las etiquetas del 0 al 9
print(imagen1a) #el resultado de la prediccion es 2, OK


#Predict for Multiple Observations (images) at Once
x1_10=logisticRegr.predict(x_test[0:10])
print("10 predicciones\n",x1_10)

print("Valor real de 'y'\n",y_test[0:10])

#Make predictions on entire test data
predictions = logisticRegr.predict(x_test)

#Setp 5. Despliegue de los resultados
# Use score method to get accuracy of model
#a) solo scores
score = logisticRegr.score(x_test, y_test)
print("Score: \n",score)

#Confusion Matrix
cm = metrics.confusion_matrix(y_test, predictions)
#matriz de confusion mostrando errores
print(cm)
#
# b) Matriz de confusion utilizando seaborn
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

#c) utilizando maplotlib
plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape

for x in range(width):  #xrange en python 2, range en python3
   for y in range(height):
     plt.annotate(str(cm[x][y]), xy=(y, x), 
     horizontalalignment='center',
     verticalalignment='center')

#Finally, you can get the report on classification as a string or dictionary 
#with classification_report():
plt.show()

print(classification_report(y_test, predictions))
