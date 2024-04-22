import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

global_neighbors = 7

# Cargar el CSV
df = pd.read_csv("diabetes.csv")

# Obtener características y objetivo de los datos originales
X_orig = df.iloc[:, :-1].values
y_orig = df.iloc[:, -1].values

# Dividir los datos originales en conjunto de entrenamiento y conjunto de prueba
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_orig, y_orig, test_size=0.3, random_state=0)

# Entrenar y evaluar clasificador KNN para datos originales
knn = KNeighborsClassifier(n_neighbors=global_neighbors)
knn.fit(X_train_orig, y_train_orig)
y_pred_orig = knn.predict(X_test_orig)
accuracy_orig = metrics.accuracy_score(y_test_orig, y_pred_orig)
cnf_matrix_orig = metrics.confusion_matrix(y_test_orig, y_pred_orig)

# Calcular métricas de evaluación para datos originales
precision_micro_orig = precision_score(y_test_orig, y_pred_orig, average='micro')
recall_micro_orig = recall_score(y_test_orig, y_pred_orig, average='micro')
f1_micro_orig = f1_score(y_test_orig, y_pred_orig, average='micro')
precision_macro_orig = precision_score(y_test_orig, y_pred_orig, average='macro')
recall_macro_orig = recall_score(y_test_orig, y_pred_orig, average='macro')
f1_macro_orig = f1_score(y_test_orig, y_pred_orig, average='macro')
precision_weighted_orig = precision_score(y_test_orig, y_pred_orig, average='weighted')
recall_weighted_orig = recall_score(y_test_orig, y_pred_orig, average='weighted')
f1_weighted_orig = f1_score(y_test_orig, y_pred_orig, average='weighted')
classification_report_orig = classification_report(y_test_orig, y_pred_orig)

# Imprimir métricas de evaluación para datos originales
print("Resultados para los datos originales:")
print('\nAccuracy: {:.2f}\n'.format(accuracy_orig))
print('Micro Precision: {:.2f}'.format(precision_micro_orig))
print('Micro Recall: {:.2f}'.format(recall_micro_orig))
print('Micro F1-score: {:.2f}\n'.format(f1_micro_orig))
print('Macro Precision: {:.2f}'.format(precision_macro_orig))
print('Macro Recall: {:.2f}'.format(recall_macro_orig))
print('Macro F1-score: {:.2f}\n'.format(f1_macro_orig))
print('Weighted Precision: {:.2f}'.format(precision_weighted_orig))
print('Weighted Recall: {:.2f}'.format(recall_weighted_orig))
print('Weighted F1-score: {:.2f}'.format(f1_weighted_orig))
print("Confusion Matrix:")
print(cnf_matrix_orig)
print("\n")

# Eliminar filas donde todas las columnas, excepto 'Pregnancies', son cero
df_no_zeros = df.loc[(df.iloc[:, 1:-1] != 0).any(axis=1)]

# Obtener características y objetivo para los datos sin ceros
X_no_zeros = df_no_zeros.iloc[:, :-1].values
y_no_zeros = df_no_zeros.iloc[:, -1].values

# Dividir los datos sin ceros en conjunto de entrenamiento y conjunto de prueba
X_train_no_zeros, X_test_no_zeros, y_train_no_zeros, y_test_no_zeros = train_test_split(X_no_zeros, y_no_zeros, test_size=0.3, random_state=42)

# Entrenar y evaluar clasificador KNN para datos sin ceros
knn.fit(X_train_no_zeros, y_train_no_zeros)
y_pred_no_zeros = knn.predict(X_test_no_zeros)
accuracy_no_zeros = metrics.accuracy_score(y_test_no_zeros, y_pred_no_zeros)
cnf_matrix_no_zeros = metrics.confusion_matrix(y_test_no_zeros, y_pred_no_zeros)

# Calcular métricas de evaluación para datos sin ceros
precision_micro_no_zeros = precision_score(y_test_no_zeros, y_pred_no_zeros, average='micro')
recall_micro_no_zeros = recall_score(y_test_no_zeros, y_pred_no_zeros, average='micro')
f1_micro_no_zeros = f1_score(y_test_no_zeros, y_pred_no_zeros, average='micro')
precision_macro_no_zeros = precision_score(y_test_no_zeros, y_pred_no_zeros, average='macro')
recall_macro_no_zeros = recall_score(y_test_no_zeros, y_pred_no_zeros, average='macro')
f1_macro_no_zeros = f1_score(y_test_no_zeros, y_pred_no_zeros, average='macro')
precision_weighted_no_zeros = precision_score(y_test_no_zeros, y_pred_no_zeros, average='weighted')
recall_weighted_no_zeros = recall_score(y_test_no_zeros, y_pred_no_zeros, average='weighted')
f1_weighted_no_zeros = f1_score(y_test_no_zeros, y_pred_no_zeros, average='weighted')
classification_report_no_zeros = classification_report(y_test_no_zeros, y_pred_no_zeros)

# Imprimir métricas de evaluación para datos sin ceros
print("Resultados para los datos sin ceros:")
print('\nAccuracy: {:.2f}\n'.format(accuracy_no_zeros))
print('Micro Precision: {:.2f}'.format(precision_micro_no_zeros))
print('Micro Recall: {:.2f}'.format(recall_micro_no_zeros))
print('Micro F1-score: {:.2f}\n'.format(f1_micro_no_zeros))
print('Macro Precision: {:.2f}'.format(precision_macro_no_zeros))
print('Macro Recall: {:.2f}'.format(recall_macro_no_zeros))
print('Macro F1-score: {:.2f}\n'.format(f1_macro_no_zeros))
print('Weighted Precision: {:.2f}'.format(precision_weighted_no_zeros))
print('Weighted Recall: {:.2f}'.format(recall_weighted_no_zeros))
print('Weighted F1-score: {:.2f}'.format(f1_weighted_no_zeros))
print("Confusion Matrix:")
print(cnf_matrix_no_zeros)
print("\n")

# Calcular el promedio de cada columna, excluyendo 'Pregnancies'
column_means = df.iloc[:, 1:-1].mean()

# Reemplazar los valores en cero por el promedio de su respectiva columna
for col in df.columns[1:-1]:
    df[col] = df[col].replace(0, column_means[col])

# Eliminar filas donde todas las columnas, excepto 'Pregnancies', son cero después de ajustar en promedio
df_adjusted = df.loc[(df.iloc[:, 1:-1] != 0).any(axis=1)]

# Obtener características y objetivo para los datos ajustados en promedio
X_adjusted = df_adjusted.iloc[:, :-1].values
y_adjusted = df_adjusted.iloc[:, -1].values

# Dividir los datos ajustados en promedio en conjunto de entrenamiento y conjunto de prueba
X_train_adjusted, X_test_adjusted, y_train_adjusted, y_test_adjusted = train_test_split(X_adjusted, y_adjusted, test_size=0.3, random_state=42)

# Entrenar y evaluar clasificador KNN para datos ajustados en promedio
knn.fit(X_train_adjusted, y_train_adjusted)
y_pred_adjusted = knn.predict(X_test_adjusted)
accuracy_adjusted = metrics.accuracy_score(y_test_adjusted, y_pred_adjusted)
cnf_matrix_adjusted = metrics.confusion_matrix(y_test_adjusted, y_pred_adjusted)

# Calcular métricas de evaluación para datos ajustados en promedio
precision_micro_adjusted = precision_score(y_test_adjusted, y_pred_adjusted, average='micro')
recall_micro_adjusted = recall_score(y_test_adjusted, y_pred_adjusted, average='micro')
f1_micro_adjusted = f1_score(y_test_adjusted, y_pred_adjusted, average='micro')
precision_macro_adjusted = precision_score(y_test_adjusted, y_pred_adjusted, average='macro')
recall_macro_adjusted = recall_score(y_test_adjusted, y_pred_adjusted, average='macro')
f1_macro_adjusted = f1_score(y_test_adjusted, y_pred_adjusted, average='macro')
precision_weighted_adjusted = precision_score(y_test_adjusted, y_pred_adjusted, average='weighted')
recall_weighted_adjusted = recall_score(y_test_adjusted, y_pred_adjusted, average='weighted')
f1_weighted_adjusted = f1_score(y_test_adjusted, y_pred_adjusted, average='weighted')
classification_report_adjusted = classification_report(y_test_adjusted, y_pred_adjusted)

# Imprimir métricas de evaluación para datos ajustados en promedio
print("Resultados para los datos ajustados en promedio:")
print('\nAccuracy: {:.2f}\n'.format(accuracy_adjusted))
print('Micro Precision: {:.2f}'.format(precision_micro_adjusted))
print('Micro Recall: {:.2f}'.format(recall_micro_adjusted))
print('Micro F1-score: {:.2f}\n'.format(f1_micro_adjusted))
print('Macro Precision: {:.2f}'.format(precision_macro_adjusted))
print('Macro Recall: {:.2f}'.format(recall_macro_adjusted))
print('Macro F1-score: {:.2f}\n'.format(f1_macro_adjusted))
print('Weighted Precision: {:.2f}'.format(precision_weighted_adjusted))
print('Weighted Recall: {:.2f}'.format(recall_weighted_adjusted))
print('Weighted F1-score: {:.2f}'.format(f1_weighted_adjusted))
print("Confusion Matrix:")
print(cnf_matrix_adjusted)

print("\n")
print("original")
print('Accuracy: {:.2f}'.format(accuracy_orig))
print(cnf_matrix_orig)
print("no zeros")
print('Accuracy: {:.2f}'.format(accuracy_no_zeros))
print(cnf_matrix_no_zeros)
print("adjusted")
print('Accuracy: {:.2f}'.format(accuracy_adjusted))
print(cnf_matrix_adjusted)
