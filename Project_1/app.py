import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load('best_model.pkl')

# Interface Streamlit
st.title("Credit_Evaluation")

st.write("""
### Entrez les caractéristiques pour obtenir une prédiction :
""")

# Formulaire pour les entrées utilisateur
input_features = []
for i in range(4):  # Supposons que le modèle utilise 4 caractéristiques
    value = st.number_input(f"Caractéristique {i+1}", format="%.2f")
    input_features.append(value)

# Prédiction
if st.button("Prediction"):
    prediction = model.predict([input_features])
    st.success(f"La classe prédite est : {prediction[0]}")
