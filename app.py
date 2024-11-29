import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("best_model.pkl")  # Utilisez un chemin relatif

# Interface Streamlit
st.title("Credit Evaluation")

st.write("""
### Enter the characteristics to obtain a prediction:
""")

# Liste des caractéristiques et leurs contraintes
feature_constraints = {
    # "ID": {"type": int, "min_id": 100000, "max_id": 999999},  # Noms propres pour ID
    "CreditScore": {"type": int, "min_score": 300, "max_score": 850},  # Noms propres pour CreditScore
    "PaymentDelays": {"type": int, "min_delay": 0, "max_delay": 12},  # Noms propres pour PaymentDelays
    "EmployedMonths": {"type": int, "min_months": 0, "max_months": 480},  # Noms propres pour EmployedMonths
    "DebtRatio": {"type": float, "min_ratio": 0.0, "max_ratio": 5.0},  # Noms propres pour DebtRatio
    "CreditAmount": {"type": float, "min_amount": 0.0, "max_amount": 1000000},  # Noms propres pour CreditAmount
    "Liquidity": {"type": float, "min_liquidity": 0.0, "max_liquidity": 100000},  # Noms propres pour Liquidity
    "CreditLines": {"type": int, "min_lines": 1, "max_lines": 10}  # Noms propres pour CreditLines
}

# Dictionnaire pour stocker les entrées utilisateur
input_features = {}

# Collecte des valeurs des caractéristiques via des inputs numériques avec des validations
for feature, constraints in feature_constraints.items():
    # Récupérer le type de la caractéristique (int ou float)
    input_type = constraints["type"]

    # Récupérer les clés de limites spécifiques à cette caractéristique
    min_key = next(k for k in constraints if k.startswith("min_"))
    max_key = next(k for k in constraints if k.startswith("max_"))
    min_value = constraints[min_key]
    max_value = constraints[max_key]
    
    # Ajouter un champ adapté au type
    if input_type == int:
        input_features[feature] = st.number_input(
            f"{feature} ({min_value} - {max_value})",
            min_value=int(min_value),
            max_value=int(max_value),
            step=1,  # Incrément pour les entiers
            format="%d"  # Format pour les entiers
        )
    elif input_type == float:
        input_features[feature] = st.number_input(
            f"{feature} ({min_value} - {max_value})",
            min_value=float(min_value),
            max_value=float(max_value),
            step=0.01,  # Incrément pour les floats
            format="%.2f"  # Format pour les floats
        )

# Prédiction
if st.button("Prediction"):
    # Préparer les données d'entrée sous forme de tableau
    input_array = np.array(list(input_features.values())).reshape(1, -1)

    # Effectuer la prédiction avec le modèle chargé
    prediction = model.predict(input_array)

    # Affichage du résultat
    st.success(f"The predicted class is: {prediction[0]}")
