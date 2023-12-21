import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


df = sklearn.datasets.load_breast_cancer()

data_frame = pd.DataFrame(df.data, columns = df.feature_names)
print(data_frame)

data_frame.head()

data_frame['target'] = df.target

print(data_frame)
print(data_frame.shape)

data_frame.info()
print(data_frame.describe())

print(data_frame['target'].value_counts())
print(data_frame.groupby('target').mean())

x = data_frame.drop(columns='target', axis=1)
y = data_frame['target']
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print(x.shape,x_train.shape,x_test.shape)

scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,),),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x_train_std,y_train, validation_split=0.1, epochs = 50)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

loss, accuracy = model.evaluate(x_test_std,y_test)
print(x_test_std.shape)
print(x_test_std[0])

y_pred = model.predict(x_test_std)
print(y_pred.shape)
print(y_pred[0])

y_pred_label = [np.argmax(i) for i in y_pred]
print(y_pred_label)

