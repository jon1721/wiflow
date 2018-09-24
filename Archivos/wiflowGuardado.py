# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import warnings
#
# warnings.filterwarnings("ignore")
# print(tf.__version__)
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

archivo = "test.csv"
df = pd.read_csv(archivo)
#print (df)
pr = pd.DataFrame(df, columns=['0', '1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
#print (pr)
xy = pd.DataFrame(df, columns = ['x', 'y'])
#print (xy)
atributos = pr[pr.columns[:20]].as_matrix()
#print (atributos)
etiquetas = xy[xy.columns[:2]].as_matrix()
#print(etiquetas)
atributos_data = np.array(atributos, "float32")
etiquetas_data = np.array(etiquetas, "float32")


predecir = atributos_data
esperado = etiquetas_data

modeloGuardado = 'model_final'

modelo = modeloGuardado + '.json'
pesos = modeloGuardado + '.h5'
 
# cargar json y crear el modelo
json_file = open(modelo, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = K.models.model_from_json(loaded_model_json)
# cargar pesos al nuevo modelo
loaded_model.load_weights(pesos)
print("Cargado modelo desde disco.")
 
# optimizador = tf.train.RMSPropOptimizer(0.001)
optimizador = K.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Compilar modelo cargado y listo para usar.
loaded_model.compile(loss='mean_squared_error', optimizer=optimizador)
loaded_model.summary()

print("Predicción con el modelo cargado")
loaded_resultado = loaded_model.predict(predecir)
#print (loaded_resultado)
print("-------------------------------------")


print("MAE--------------------------------->")

mae = np.absolute(esperado - loaded_resultado)
cantidad = mae.size/mae.ndim
# print(cantidad)
mae_final = np.sum(mae) / cantidad
print("mae: ", mae_final)
print("MAE---------------------------------<")

esperado_array = np.array(esperado, "float32")
resultado_array = np.array(loaded_resultado, "float32")
mae_array = np.array(mae, "float32")
np.savetxt("esperado.csv", esperado_array, delimiter=",")
np.savetxt("resultado.csv", loaded_resultado, delimiter=",")
np.savetxt("mae.csv", mae, delimiter=",")

trp = esperado.transpose()
rtp = loaded_resultado.transpose()

np.set_printoptions(precision=2, suppress=True)
print(rtp)
print (trp)

plt.xlabel('Coord X')
plt.ylabel('Coord Y')
plt.plot(trp[0], trp[1], 'ro')
plt.plot(rtp[0], rtp[1], 'g^')
plt.axis([-4, 14, -4, 16])
plt.show()


print("-------------------------------------")

