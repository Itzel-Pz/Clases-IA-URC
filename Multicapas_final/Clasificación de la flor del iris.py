from sklearn.datasets import load_iris
iris = load_iris()

# aquí cargo el dataset iris que ya viene incluido en sklearn, y lo guardo en la variable iris
print(iris.keys())
# imprimo las llaves del dataset para ver qué contiene (data, target, feature_names, etc.)

print(iris.DESCR)
# imprimo la descripción del dataset para conocer qué variables tiene, cuántos datos son, etc.


import pandas as pd

# convierto los datos numéricos del iris a un DataFrame de pandas para poder visualizarlos mejor
iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
iris_df

# describo el dataframe para ver media, desviación estándar, mínimo, máximo, etc.
iris_df.describe()


# X son las variables independientes, es decir las características de cada flor
X = iris_df

# y son las etiquetas (las clases). Las convierto a variables dummy (one-hot) porque la rede neuronal
# trabaja mejor así que con números enteros 0,1,2
y = pd.get_dummies(iris.target).values


from sklearn.model_selection import train_test_split

# separo los datos: 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)


from sklearn.preprocessing import StandardScaler

# escalo los datos (normalizo). Esto es importante para redes neuronales,
# para que todas las variables estén en la misma escala y el entrenamiento sea más estable
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# creo el modelo secuencial
# capa de entrada -> 64 neuronas, activación ReLU
# capa oculta -> 32 neuronas, ReLU
# capa de salida -> 3 neuronas, softmax (porque son 3 clases)
modelo = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax'),
])


from tensorflow.keras.optimizers import Adam

# defino tasa de aprendizaje y creo el optimizador Adam
learning_rate = 0.001
adam_optimizer = Adam(learning_rate=learning_rate)


# aquí compilo el modelo, le digo qué optimizador usar, qué función de pérdida
# y qué métrica quiero monitorear
modelo.compile(
    optimizer=adam_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# entreno el modelo durante 10 épocas
history = modelo.fit(
    X_train, y_train,
    epochs=10, batch_size=1,
    validation_data=(X_test, y_test)
)


import matplotlib.pyplot as plt

# grafico la pérdida (loss) durante el entrenamiento y validación
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Función de pérdida durante el entrenamiento')
plt.show()

# evalúo el modelo con el conjunto de prueba
loss, accuracy = modelo.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')


predictions = modelo.predict(X_test)

# obtengo el índice de la clase predicha (0,1 o 2)
predicted_classes = np.argmax(predictions, axis=1)

# obtengo el índice de la clase real
actual_classes = np.argmax(y_test, axis=1)

# creo un DataFrame para comparar resultados reales vs predichos
comparison = pd.DataFrame({'Actual Class Index': actual_classes, 'Predicted Class Index': predicted_classes})
# agrego nombre real y predicho de la flor
comparison['Actual Flower'] = [iris.target_names[i] for i in actual_classes]
comparison['Predicted Flower'] = [iris.target_names[i] for i in predicted_classes]
# reordeno columnas
comparison = comparison[['Actual Class Index', 'Actual Flower', 'Predicted Class Index', 'Predicted Flower']]
print(comparison.head())


import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

# otra vez predigo (es lo mismo que antes)
y_pred = modelo.predict(X_test)

# convierto probabilidades a clases
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# calculo matriz de confusión
cm = confusion_matrix(y_test_classes, y_pred_classes)

# calculo sensibilidad (recall) por clase
sensitivity = recall_score(y_test_classes, y_pred_classes, average=None)

# muestro matriz de confusión como imagen
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')

# imprimo sensibilidad por clase
print('Sensitivity (Recall) for each class:')
for i in range(3):
    print(f'Class {i}: {sensitivity[i]}')
