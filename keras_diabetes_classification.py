from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import tkinter as tk
import numpy as np


def train_the_model():
    # seed aleatoria para reproducibilidad
    np.random.seed(7)
    
    # carga dataset de casos de diabetes en la india, últimos 5 años de historia clínica
    dataset = np.loadtxt("prima-indians-diabetes.csv", delimiter=",")
    
    # split into input (X) and output (Y) variables, splitting csv data
    X = dataset[:,0:8]
    Y = dataset[:,8]
    # Se dividen los datos de entrenamiento(80%) y campos de test(20%)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Se crea modelo,  se agregan layers densos para cada funcion de activacion
    model = Sequential()
    model.add(Dense(15, input_dim=8, activation='relu')) # input layer requiere input_dim param
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='sigmoid')) # sigmoid en lugar de relu para la probabilidad final entre 0 y 1
    
    # Compila el modelo, se usa el optimizer adam
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    # Se llama la funcion fit, para testear el modelo
    model.fit(x_train, y_train, epochs = 1000, batch_size=20, validation_data=(x_test, y_test))
    
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)    
    return model

def test_with_numpy(model, test):
    predictions = model.predict(test)
    print('predictions shape:', predictions)    
    return

def button_click(inputs, model, root):
    predictions = model.predict(inputs)
    print('predictions shape:', predictions)
    #print('exactitud = ', model.accuracy())
    value = predictions[0][0]
    percentaje = value * 100
    texto = 'Tu probabilidad es de: ' + str(round(percentaje, 2)) + '%'
    print(texto)
    label_result = tk.Label(root, text=texto, font = ('bold', 20),anchor='w')
    label_result.place(x=20, y=440)
    return

def create_form(model):
    root = tk.Tk()
    root.geometry('600x500')
    root.title('Formulario para predcir la diabetes')
    
    label_title = tk.Label(root, text="Ingresa los siguientes datos: ", font = ('bold', 20),anchor='w')
    label_title.place(x=20, y=10)
    
    #num Embarazos
    label_0 = tk.Label(root, text="Numero de embarazos: ", anchor='w')
    label_0.place(x=20, y=80)
    entry_0 = tk.Entry(root)
    entry_0.place(x=250, y=80)
    #Concentracion de glucosa 0 - 200
    label_1 = tk.Label(root, text="Concentración de glucosa: ", anchor='w')
    label_1.place(x=20, y=120)
    entry_1 = tk.Entry(root)
    entry_1.place(x=250, y=120)
    #Presion sanguinea 0 - 100
    label_2 = tk.Label(root, text="Presión sanguínea diastólica: ", anchor='w')
    label_2.place(x=20, y=160)
    entry_2 = tk.Entry(root)
    entry_2.place(x=250, y=160)
    #Grosura de la piel 0 - 50
    label_3 = tk.Label(root, text="Grosura del pliegue cutáneo en triceps: ", anchor='w')
    label_3.place(x=20, y=200)
    entry_3 = tk.Entry(root)
    entry_3.place(x=250, y=200)
    #Insulina en la sangre 20 - 120
    label_4 = tk.Label(root, text="Suero de insulina en la sangre: ", anchor='w')
    label_4.place(x=20, y=240)
    entry_4 = tk.Entry(root)
    entry_4.place(x=250, y=240)
    #IMC 0.0 - 50.0
    label_5 = tk.Label(root, text="-Indice de masa corporal (IMC): ", anchor='w')
    label_5.place(x=20, y=280)
    entry_5 = tk.Entry(root)
    entry_5.place(x=250, y=280)
    #Funcion de pedrigree 0- 60
    label_6 = tk.Label(root, text="-Función de pedigree de diabetes: ", anchor='w')
    label_6.place(x=20, y=320)
    entry_6 = tk.Entry(root)
    entry_6.place(x=250, y=320)
    #Edad
    label_7 = tk.Label(root, text="Edad: ", anchor='w')
    label_7.place(x=20, y=360)
    entry_7 = tk.Entry(root)
    entry_7.place(x=250, y=360)
    #Button
    button = tk.Button(root, text='Calcular', bg='brown', fg='white', command= lambda: button_click( 
        np.array([[
            float(entry_0.get()),
            float(entry_1.get()),
            float(entry_2.get()),
            float(entry_3.get()),
            float(entry_4.get()),
            float(entry_5.get()),
            float(entry_6.get()),
            float(entry_7.get())
        ]]),
        model,
        root
    ))
    button.place(x=250, y=400)
    root.mainloop()
    
    return root

#prueba = np.array([[0,137,84,27,0,27.3,0.231,59]])
model = train_the_model()
form = create_form(model)


#test_with_numpy(model, prueba)



    