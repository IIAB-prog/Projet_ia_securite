import os

# Désactivation GPU avant TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib


# Configuration CPU TensorFlow
tf.config.set_visible_devices([], 'GPU')


# ==========================================================
# CONFIGURATION PAGE
# ==========================================================

st.set_page_config(
    page_title="IDS2025 - Détection Intrusions",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================================
# CHARGEMENT MODELE (AU BESOIN)
# ==========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "ids_model.keras"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    encoder = joblib.load(
        "encoder.pkl"
    )

    return model, scaler, encoder



# ==========================================================
# TITRE
# ==========================================================

st.title("🛡️ IDS2025")
st.subheader(
    "Détection intelligente des intrusions réseau par Deep Learning"
)

st.divider()



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

    st.write(
        """
Cette application permet la détection automatique
des intrusions réseau grâce à un modèle Deep Learning.

Fonctionnalités :

✅ Import CSV (79 caractéristiques réseau)

✅ Import Excel XLSX

✅ Classification automatique

✅ Probabilité de prédiction

✅ Export des résultats
"""
    )



# ==========================================================
# ANALYSE CSV/XLSX
# ==========================================================

elif menu == "Analyse fichier CSV/XLSX":


    st.header("📂 Analyse d'un fichier réseau")


    uploaded_file = st.file_uploader(
        "Importer un fichier CSV ou Excel",
        type=["csv", "xlsx"]
    )


    if uploaded_file is not None:


        # Taille fichier

        taille = uploaded_file.size / (1024 * 1024)

        st.info(
            f"Taille du fichier : {taille:.2f} MB"
        )


        if taille > 50:

            st.error(
                "Fichier trop volumineux (maximum 50 MB)"
            )

            st.stop()



        # Lecture fichier

        try:


            if uploaded_file.name.endswith(".csv"):


                dataframe = pd.read_csv(
                    uploaded_file,
                    low_memory=False
                )


            else:


                dataframe = pd.read_excel(
                    uploaded_file,
                    engine="openpyxl"
                )


        except Exception as e:


            st.error(
                f"Erreur lecture fichier : {e}"
            )

            st.stop()



        # Limite mémoire

        if dataframe.shape[0] > 50000:


            st.warning(
                "Le fichier contient plus de 50000 lignes. "
                "Seules les 50000 premières seront analysées."
            )


            dataframe = dataframe.head(50000)



        st.subheader(
            "Aperçu des données"
        )


        st.dataframe(
            dataframe.head()
        )


        st.write(
            f"Lignes : {dataframe.shape[0]}"
        )


        st.write(
            f"Colonnes : {dataframe.shape[1]}"
        )



        # Bouton analyse

        if st.button(
            "🚀 Lancer la détection"
        ):


            try:


                with st.spinner(
                    "Chargement du modèle IA..."
                ):


                    model, scaler, encoder = load_model()



                # Colonnes attendues

                expected_features = list(
                    scaler.feature_names_in_
                )



                missing_columns = list(
                    set(expected_features)
                    -
                    set(dataframe.columns)
                )



                if missing_columns:


                    st.error(
                        "Colonnes manquantes :"
                    )

                    st.write(
                        missing_columns
                    )

                    st.stop()



                # Réorganisation

                X = dataframe[
                    expected_features
                ]



                # Normalisation

                X_scaled = scaler.transform(
                    X
                )



                st.info(
                    "Prédiction en cours..."
                )


                # Prediction par batch

                prediction = model.predict(
                    X_scaled,
                    batch_size=32
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



                dataframe["Prediction"] = labels


                dataframe["Probabilite (%)"] = (
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



                st.subheader(
                    "Statistiques"
                )


                st.bar_chart(
                    dataframe["Prediction"]
                    .value_counts()
                )



                # Export

                resultat = dataframe.to_csv(
                    index=False
                )


                st.download_button(
                    "📥 Télécharger les résultats",
                    resultat,
                    file_name="IDS2025_resultats.csv",
                    mime="text/csv"
                )



            except Exception as e:


                st.error(
                    f"Erreur pendant analyse : {e}"
                )



# ==========================================================
# A PROPOS
# ==========================================================

elif menu == "À propos":


    st.header(
        "À propos du projet"
    )


    st.markdown(
        """
## IDS2025

Système intelligent de détection
des intrusions réseau.

### Technologies

- Python
- TensorFlow/Keras
- Streamlit
- Scikit-Learn
- Pandas

### Modèle

MLP (Multi Layer Perceptron)

### Entrées

Fichier CSV/XLSX contenant
les caractéristiques réseau.

### Sorties

- Trafic normal
- Trafic malveillant
- Probabilité
"""
    )



# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Projet Master 2 IA - Détection des Intrusions Réseau - IMA ADAM"
)
