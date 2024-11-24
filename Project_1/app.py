import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("best_model.pkl")  # Utilisez un chemin relatif

# Interface Streamlit
st.title("Credit Evaluation")

st.write("""
### Enter the characteristics to obtain a prediction :
""")

# Liste des caractéristiques et leurs contraintes
feature_constraints = {
    "ID": {"type": int, "min_value": 100000, "max_value": 999999},  # ID doit être un entier entre 100000 et 999999
    "CreditScore": {"type": int, "min_value": 300, "max_value": 850},  # Score de crédit entre 300 et 850
    "PaymentDelays": {"type": int, "min_value": 0, "max_value": 12},  # Nombre de retards de paiement (de 0 à 12)
    "EmployedMonths": {"type": int, "min_value": 0, "max_value": 480},  # Mois d'emploi (de 0 à 480 mois)
    "DebtRatio": {"type": float, "min_value": 0.0, "max_value": 5.0},  # Ratio d'endettement entre 0 et 5
    "CreditAmount": {"type": float, "min_value": 0.0, "max_value": 1000000},  # Montant de crédit (de 0 à 1 million)
    "Liquidity": {"type": float, "min_value": 0.0, "max_value": 100000},  # Liquidité (de 0 à 100000)
    "CreditLines": {"type": int, "min_value": 1, "max_value": 10}  # Nombre de lignes de crédit (de 1 à 10)
}

# Dictionnaire pour stocker les entrées utilisateur
input_features = {}

# Collecte des valeurs des caractéristiques via des inputs numériques avec des validations
for feature, constraints in feature_constraints.items():
    value = None
    while value is None:  # Demander à l'utilisateur jusqu'à ce qu'il entre une valeur valide
        # Utilisation de st.number_input sans format pour éviter le conflit de type
        if constraints['type'] == int:
            value = st.number_input(
                f"{feature} ({constraints['min_value']} - {constraints['max_value']})", 
                min_value=constraints['min_value'],
                max_value=constraints['max_value'],
                step=1  # Pour les entiers, on fixe l'incrément à 1
            )
        elif constraints['type'] == float:
            value = st.number_input(
                f"{feature} ({constraints['min_value']} - {constraints['max_value']})", 
                min_value=constraints['min_value'],
                max_value=constraints['max_value'],
                format="%.2f"  # Pour les floats, conserver le format avec deux décimales
            )
        
        # Validation si la valeur est valide
        if value < constraints['min_value'] or value > constraints['max_value']:
            st.warning(f"Veuillez entrer une valeur pour {feature} entre {constraints['min_value']} et {constraints['max_value']}.")
            value = None  # Réinitialiser la valeur si elle est en dehors de la plage

    input_features[feature] = value

# Prédiction
if st.button("Prediction"):
    # Préparer les données d'entrée sous forme de tableau
    input_array = np.array(list(input_features.values())).reshape(1, -1)

    # Effectuer la prédiction avec le modèle chargé
    prediction = model.predict(input_array)

    # Affichage du résultat
    st.success(f" The predicted class is : {prediction[0]}")
