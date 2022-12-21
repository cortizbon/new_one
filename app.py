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


