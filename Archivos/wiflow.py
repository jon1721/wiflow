import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
print(tf.__version__)


#archivo = "campoWiFi.csv"
#archivo = "compensado.csv"
#archivo = "reales2.csv"
# archivo = "training.csv"
# archivo = "dbm.csv"
# archivo = "training0.csv"
archivo = "train.csv"
epochs = 20000

df = pd.read_csv(archivo)
# print (df)
pr = pd.DataFrame(df, columns=['0', '1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
# print (pr)
xy = pd.DataFrame(df, columns=['x', 'y'])
# print (xy)
atributos = pr[pr.columns[:20]].as_matrix()
# print (atributos)
etiquetas = xy[xy.columns[:2]].as_matrix()
# print(etiquetas)
atributos_data = np.array(atributos, "float32")
etiquetas_data = np.array(etiquetas, "float32")

print("atributos_data ---------->")
print(atributos_data.shape)
print("atributos_data ----------END")

# inicializador = K.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
# inicializador = K.initializers.Zeros();
# inicializador = K.initializers.Ones()

# activacion = K.layers.Activation('softmax')
# relu = K.layers.Activation('relu')
# tanh = K.layers.Activation('tanh')

# inicializador = K.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
# inicializador = K.initializers.Zeros();
# inicializador = K.initializers.Ones()
# inicializador = K.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=11)
inicializador = 'random_normal'


model = K.Sequential()

model.add(
    K.layers.Dense(32, input_dim=20, activation='relu', kernel_initializer=inicializador, bias_initializer=inicializador))
model.add(
    K.layers.Dense(64, activation='sigmoid', kernel_initializer=inicializador, bias_initializer=inicializador))
model.add(
    K.layers.Dense(32, activation='sigmoid', kernel_initializer=inicializador, bias_initializer=inicializador))
model.add(
    K.layers.Dense(16, activation='sigmoid', kernel_initializer=inicializador, bias_initializer=inicializador))
model.add(
    K.layers.Dense(2, activation='relu', kernel_initializer=inicializador, bias_initializer=inicializador))

# optimizador = K.optimizers.SGD(lr=0.01, clipnorm=1.)
# optimizador = K.optimizers.SGD(lr=0.01, clipnorm=1.)
#optimizador = K.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# optimizador = K.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
# optimizador = K.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
optimizador = K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optimizador = K.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

tbCallBack  = K.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,  write_graph=True, write_images=False)
model.compile(loss='mse', optimizer=optimizador, metrics=['mae'])
history = model.fit(atributos_data, etiquetas_data, epochs=epochs, verbose=1, callbacks=[tbCallBack])

model.summary()

loss = history.history['loss']
mae = history.history['mean_absolute_error']

plt.xlabel('Epoch')
plt.ylabel('loss/mae')
plt.plot(history.epoch, loss, label='loss')
plt.plot(history.epoch, mae, label='mae')
plt.legend()
plt.axis([0, epochs*1.2, 0, 10])
plt.show()



scores = model.evaluate(atributos_data, etiquetas_data, verbose=1)

print("model.metrics_names")
print(model.metrics_names)
print(scores)


predecir = atributos_data
esperado = etiquetas_data

resultado = model.predict(predecir)

dif = np.subtract(esperado, resultado)
difabs = np.abs(dif)
media = np.ndarray.mean(difabs)
min = np.ndarray.min(difabs)
max = np.ndarray.max(difabs)

print("********* mae: %.2f, min: %.2f, max: %.2f" % (media, min, max) )


# gran = np.where( difabs > media )

gran = ( difabs > media ).sum()

trp = esperado.transpose()
rtp = resultado.transpose()

plt.xlabel('Coord Y')
plt.ylabel('Coord X')
plt.plot(trp[0], trp[1], 'ro')
plt.plot(rtp[0], rtp[1], 'g^')
plt.axis([-4, 12, -4, 18])
plt.show()

print("-------------------------------------")

# serializar el modelo a JSON
model_json = model.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json)
# serializar los pesos a HDF5
model.save_weights("model_final.h5")
print("Modelo Guardado!")
model.save('model_final.hdf5')
print("Modelo HDF5 Guardado!")

