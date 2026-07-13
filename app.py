import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================================
# CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="IDS2025",
    page_icon="🛡️",
    layout="wide"
)



# ==========================================================
# CHARGEMENT MODELE
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "ids_model.pkl"
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

st.title(
    "🛡️ IDS2025"
)

st.subheader(
    "Détection intelligente des intrusions réseau"
)


st.divider()



# ==========================================================
# MENU
# ==========================================================

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Accueil",
        "Analyse CSV/XLSX",
        "A propos"
    ]
)



# ==========================================================
# ACCUEIL
# ==========================================================

if menu == "Accueil":


    st.header(
        "Présentation"
    )


    st.write(
        """
Cette application utilise un modèle Deep Learning
pour classifier automatiquement les flux réseau.

Fonctionnalités :

✅ Import CSV

✅ Import Excel XLSX

✅ Analyse de plusieurs connexions

✅ Classification normale / attaque

✅ Export des résultats
"""
    )



# ==========================================================
# ANALYSE FICHIER
# ==========================================================

elif menu == "Analyse CSV/XLSX":


    st.header(
        "📂 Charger un fichier réseau"
    )


    fichier = st.file_uploader(
        "Choisir un fichier",
        type=[
            "csv",
            "xlsx"
        ]
    )



    if fichier:


        try:


            if fichier.name.endswith(".csv"):


                data = pd.read_csv(
                    fichier,
                    low_memory=False
                )


            else:


                data = pd.read_excel(
                    fichier,
                    engine="openpyxl"
                )


        except Exception as e:


            st.error(
                f"Erreur lecture fichier : {e}"
            )

            st.stop()



        st.success(
            "Fichier chargé"
        )


        st.write(
            "Dimensions :",
            data.shape
        )


        st.dataframe(
            data.head()
        )



        if st.button(
            "🚀 Analyser"
        ):


            try:


                with st.spinner(
                    "Chargement du modèle..."
                ):


                    model, scaler, encoder = load_model()



                # Colonnes modèle

                features = list(
                    scaler.feature_names_in_
                )



                missing = list(
                    set(features)
                    -
                    set(data.columns)
                )



                if missing:


                    st.error(
                        "Colonnes absentes :"
                    )


                    st.write(
                        missing
                    )


                    st.stop()



                # Garder uniquement les variables IA

                X = data[
                    features
                ]



                # Normalisation

                X_scaled = scaler.transform(
                    X
                )



                # Prediction

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



                scores = np.max(
                    prediction,
                    axis=1
                )



                data["Prediction"] = labels


                data["Probabilite"] = (
                    scores * 100
                ).round(2)



                st.success(
                    "Analyse terminée"
                )



                st.dataframe(
                    data
                )



                st.subheader(
                    "Statistiques"
                )


                st.bar_chart(
                    data["Prediction"]
                    .value_counts()
                )



                result = data.to_csv(
                    index=False
                )


                st.download_button(
                    "📥 Télécharger résultat",
                    result,
                    file_name="IDS2025_resultats.csv",
                    mime="text/csv"
                )



            except Exception as e:


                st.error(
                    f"Erreur : {e}"
                )



# ==========================================================
# A PROPOS
# ==========================================================

elif menu == "A propos":


    st.header(
        "Projet IDS2025"
    )


    st.markdown(
        """
### Technologies

- Python
- Streamlit
- Scikit-Learn
- Deep Learning

### Modèle

MLP (Multi Layer Perceptron)

### Entrée

79 caractéristiques réseau

### Sortie

- Trafic normal
- Trafic malveillant
"""
    )



st.divider()

st.caption(
    "Master 2 IA - Projet Détection des Intrusions Réseau"
)
