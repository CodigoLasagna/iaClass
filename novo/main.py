import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Cargar el CSV
df = pd.read_csv("diabetes.csv")

# Eliminar columnas vacías
#df = df.dropna(axis=1, how='all')
# Eliminar filas donde el valor es cero en todas las columnas excepto 'Pregnancies'
df = df.loc[(df.iloc[:, 1:-1] != 0).any(axis=1)]

# Obtener características y objetivo
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar el clasificador KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo
knn.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = knn.predict(X_test)

# Calcular la precisión
print("Accuracy_5:", metrics.accuracy_score(y_test, y_pred))

# Cambiar el número de vecinos a 11
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy_11:", metrics.accuracy_score(y_test, y_pred))

# Calcular la matriz de confusión
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cnf_matrix)

# Imprimir métricas de evaluación
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

# Imprimir informe de clasificación
print('\nClassification Report\n')
print(classification_report(y_test, y_pred))
