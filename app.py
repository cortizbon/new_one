import streamlit as st
import pandas as pd
import pickle
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Título
data = pd.read_csv('data.csv')

st.title("Predicción e interpretación: Titanic")

# Exploración inicial

## Head

st.dataframe(data.head())
## Describe

st.dataframe(data.describe())
## Info


buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

# Gráficas


## Bivariadas
fig, ax = plt.subplots(1, 1)

sns.boxplot(data=data, x='pclass', y='age', hue='survived', ax=ax)

st.pyplot(fig)

fig, ax = plt.subplots(1, 1)

sns.boxplot(data=data, x='sex_male', y='age', hue='survived', ax=ax)

st.pyplot(fig)
# Modelo

model = pickle.load(open("model.pickle", 'rb'))

## Interpretación del modelo


# Predicción

col1, col2 = st.cols(2)

## Columna 1

with col1:
  sex = st.selectbox("Sexo", ("F", "M"))
  if sex == "F":
    sex = 0
  else:
    sex = 1
  age = st.slider("Edad", 0, 95)
  fare = st.slider("Disposición a pagar", 0, 800)
  
## Columna 2

with col2:
  pclass = st.selectbox("Clase", (1, 2, 3))
  sibsp = st.slider("Número de hermanos/as en el viaje", 0, 8)
  parch = st.slider("Número de padres y/o hijos", 0, 8)
## Resultado

if st.button('Predecir'):
   val = model.predict_proba(np.array([[pclass, age, sibsp, parch, fare, sex]]))
   st.text("La probabilidad de sobrevivir es de {round(val[0][1], 2)%}")
