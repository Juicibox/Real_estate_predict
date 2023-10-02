import streamlit as st
import pickle
import pandas as pd
from pycaret.regression import *  #pip install pycaret
import xgboost

st.set_page_config(page_title="Price_House", page_icon="🏠")

# Definir modelos por ciudad
modelos = {
    "Bogota": "modelos/modelo_final_bogo",
    "Cali": "modelo_final_cal",
    "Cartagena": "modelo_final_cartagena",
    "Medellin": "modelo_final_medellin"
}

# Definir listas de barrios por ciudad
barrios_ciudad_A = ['Cedritos', 'Fontibon', 'Colina Campestre', 'Santa Barbara', 'San Jose De Bavaria', 'Chico Norte', 'Usaquen','Chapinero', 'Engativá', 'Santa Paula', 'Los Rosales', 'Mazuren', 
                    'Calleja', 'Suba', 'Santa Barbara Occidental', 'Batan', 'Refugio', 'Pasadena', 'Molinos Norte', 'Lagos De Cordoba', 'Tintal', 'Chico Navarra', 'Bellavista']

barrios_ciudad_B = ["Pance", "Valle Del Lili", "Santa Teresita", "Ciudad Jardin", "Ciudad Campestre", "Prados Del Norte", "La Flora", "El Ingenio",
                    "Bellavista", "Caney", "Aguacatal","Bochalema", "La Hacienda", "Santa Rita", "Santa Anita", "Juanambu", "Villa Fatima", "El Refugio"]

barrios_ciudad_C = ["Manga","Crespo","Boquilla","Castillo Grande","Pie De La Popa","Zona Norte","Laguito","Cabrero","Boca Grande",
                    "Recreo","Torices","Marbella","Concepcion","Providencia","Alameda La Victoria","Getsemani","Barcelona De Indias"]

barrios_ciudad_D =  ['Poblado', 'Belen', 'Laureles', 'Lorena', 'Patio Bonito', 'Calasanz', 'San Lucas', 'Candelaria', 'Robledo', 'Conquistadores', 'Via Las Palmas','Belen Rosales', 'Simon Bolivar', 
                     'Castellana', 'Las Palmas', 'Pilarica', 'San Diego', 'Altos Del Poblado', 'Castropol', 'San Antonio De Prado']


# Añadir una barra lateral (sidebar)
st.sidebar.title("Configuración")

# Agregar una lista desplegable para elegir la ciudad
ciudad = st.sidebar.selectbox("Elegir ciudad:", list(modelos.keys()))



# Asignar el modelo correspondiente a la ciudad seleccionada
modelo = load_model(modelos[ciudad])

# Asignar la lista de barrios correspondiente a la ciudad seleccionada
if ciudad == "Bogota":
    barrios = st.sidebar.selectbox("Elegir zona:", barrios_ciudad_A)
    with open('lista_colum_bogo.pkl', 'rb') as f:
        mi_lista = pickle.load(f)
elif ciudad == "Cali":
    barrios = st.sidebar.selectbox("Elegir zona:", barrios_ciudad_B)
    with open('lista_colum_cali.pkl', 'rb') as f:
        mi_lista = pickle.load(f)
elif ciudad == "Cartagena":
    barrios = st.sidebar.selectbox("Elegir zona:", barrios_ciudad_C)
    with open('lista_colum_cartage.pkl', 'rb') as f:
        mi_lista = pickle.load(f)
elif ciudad == "Medellin":
    barrios = st.sidebar.selectbox("Elegir zona:", barrios_ciudad_D)
    with open('lista_colum_medellin.pkl', 'rb') as f:
        mi_lista = pickle.load(f)

# Resto del código
if st.button('Volver a proyectos'):
    st.markdown('<a href="https://juicibox.github.io/proyectos.html" target="_self">Click</a>', unsafe_allow_html=True)

st.title("Predicción Valor de Vivienda 🏠")

st.write(f"El modelo estima los precios de vivienda para la ciudad de {ciudad}.")

# funcion para seleccionar barrios
def seleccionar_barrio(barrio_seleccionado, columnas_barrios):
    valores_barrios = {col: 0 for col in columnas_barrios}
    valores_barrios[barrio_seleccionado] = 1
    return valores_barrios

barrio_seleccionado = None

# Verificar si se seleccionó un barrio y asignarlo a la variable
if barrios:
    barrio_seleccionado = f'barrio_{barrios}'
    valores_barrios  = seleccionar_barrio(barrio_seleccionado, mi_lista)

area = st.slider("Área", 30, 1000, 80)
habitacion = st.slider("Habitaciones", 1, 6, 2)
bano = st.slider("Baños", 1, 6, 1)
parqueadero = st.slider("Parqueaderos", 0, 5, 1)

valores_seleccionados = {
        "area": area,
        "habitaciones": habitacion,
        "garajes": parqueadero,
        "banos": bano
    }

ok = st.button("Calcular Precio")
if ok:

    valores_combinados = {**valores_seleccionados, **valores_barrios}
    datos = pd.DataFrame([valores_combinados])
    prediccion = predict_model(modelo, data=datos)

    num = int(prediccion.iloc[:, -1])
    st.subheader(f"El precio estimado de la vivienda es: {num:,d} millones de pesos.")
