import os
# Désactiver GPU CUDA avant le chargement TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import joblib


# ==========================================================
# CONFIGURATION DE LA PAGE
# ==========================================================

st.set_page_config(
    page_title="IDS2025 - Détection d'Intrusions",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================================
# CHARGEMENT DU MODELE
# ==========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model("ids_model.keras")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")

    return model, scaler, encoder


model, scaler, encoder = load_model()



# ==========================================================
# TITRE
# ==========================================================

st.title("🛡️ IDS2025")
st.subheader(
    "Détection intelligente des intrusions réseau par Deep Learning"
)

st.markdown("---")



# ==========================================================
# MENU
# ==========================================================

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Accueil",
        "Analyse fichier CSV/XLSX",
        "À propos"
    ]
)



# ==========================================================
# ACCUEIL
# ==========================================================

if menu == "Accueil":

    st.header("Présentation")

    st.write("""
Cette application permet de détecter automatiquement
les intrusions réseau grâce à un modèle Deep Learning.

Fonctionnalités :

✅ Importation CSV(79 colonnes necessaires, 0-n lignes)  

✅ Importation Excel XLSX (79 colonnes necessaires, 0-n lignes)

✅ Classification automatique des connexions réseau  

✅ Probabilité de prédiction  

✅ Export des résultats
""")



# ==========================================================
# ANALYSE CSV / XLSX
# ==========================================================

elif menu == "Analyse fichier CSV/XLSX":

    st.header("📂 Analyse d'un fichier réseau")

    uploaded_file = st.file_uploader(
        "Importer un fichier CSV ou Excel",
        type=["csv", "xlsx"]
    )


    if uploaded_file is not None:


        # Lecture fichier

        if uploaded_file.name.endswith(".csv"):

            dataframe = pd.read_csv(uploaded_file)

        else:

            dataframe = pd.read_excel(uploaded_file)



        st.subheader("Aperçu des données")

        st.dataframe(dataframe.head())



        st.write(
            f"Nombre de lignes : {dataframe.shape[0]}"
        )

        st.write(
            f"Nombre de colonnes : {dataframe.shape[1]}"
        )



        if st.button("🚀 Lancer la détection"):


            try:

                # Colonnes attendues par le scaler

                expected_features = list(
                    scaler.feature_names_in_
                )


                missing_columns = list(
                    set(expected_features)
                    - set(dataframe.columns)
                )


                if missing_columns:

                    st.error(
                        "Colonnes manquantes dans le fichier :"
                    )

                    st.write(missing_columns)

                    st.stop()



                # Réorganisation des colonnes

                X = dataframe[expected_features]


                # Normalisation

                X_scaled = scaler.transform(X)



                # Prédiction

                prediction = model.predict(
                    X_scaled
                )


                classes = np.argmax(
                    prediction,
                    axis=1
                )


                labels = encoder.inverse_transform(
                    classes
                )


                probabilites = np.max(
                    prediction,
                    axis=1
                )



                # Ajout résultats

                dataframe["Prediction"] = labels

                dataframe["Probabilite"] = (
                    probabilites * 100
                ).round(2)



                st.success(
                    "✅ Analyse terminée"
                )


                st.subheader(
                    "Résultats"
                )


                st.dataframe(
                    dataframe
                )



                # Statistiques

                st.subheader(
                    "Résumé"
                )


                st.bar_chart(
                    dataframe["Prediction"]
                    .value_counts()
                )



                # Export

                csv = dataframe.to_csv(
                    index=False
                )


                st.download_button(
                    label="📥 Télécharger les résultats",
                    data=csv,
                    file_name="IDS2025_resultats.csv",
                    mime="text/csv"
                )


            except Exception as e:

                st.error(
                    f"Erreur pendant l'analyse : {e}"
                )



# ==========================================================
# A PROPOS
# ==========================================================

elif menu == "À propos":

    st.header("À propos du projet")


    st.write("""
## IDS2025

Système intelligent de détection
des intrusions réseau.

### Technologies

- Python
- TensorFlow / Keras
- Streamlit
- Scikit-Learn
- Pandas
- NumPy

### Modèle

MLP (Multi Layer Perceptron)

### Entrée

Fichier CSV ou XLSX contenant
les caractéristiques réseau.

### Sortie

Classification :
- Trafic normal
- Trafic malveillant
""")



# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Projet Master 2 IA- U_AUBEN - Détection des Intrusions Réseau - By IMA ADAM"
)
