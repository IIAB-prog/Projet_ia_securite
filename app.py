import os

# ================================
# CONFIGURATION CPU RENDER
# ================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib



# ================================
# CONFIGURATION STREAMLIT
# ================================

st.set_page_config(
    page_title="IDS2025 - Détection Intrusion",
    page_icon="🛡️",
    layout="wide"
)



# ================================
# CHARGEMENT MODELE
# ================================

@st.cache_resource(show_spinner=True)
def load_model():

    with tf.device("/CPU:0"):

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



model, scaler, encoder = load_model()



# ================================
# TITRE
# ================================

st.title("🛡️ IDS2025")
st.subheader(
    "Détection intelligente des intrusions réseau par Deep Learning"
)


st.divider()



# ================================
# MENU
# ================================

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Accueil",
        "Analyse CSV/XLSX",
        "A propos"
    ]
)



# ================================
# ACCUEIL
# ================================

if menu == "Accueil":


    st.header("Présentation")


    st.write(
    """
    Cette application utilise un modèle Deep Learning
    pour détecter automatiquement les intrusions réseau.

    Fonctionnalités :

    ✅ Import CSV  
    ✅ Import Excel XLSX  
    ✅ Analyse automatique  
    ✅ Classification trafic normal/malveillant  
    ✅ Export des résultats
    """
    )



# ================================
# ANALYSE FICHIER
# ================================

elif menu == "Analyse CSV/XLSX":


    st.header("📂 Charger un fichier réseau")


    uploaded_file = st.file_uploader(
        "Choisir un fichier",
        type=["csv","xlsx"]
    )



    if uploaded_file:


        # Limite 50 Mo

        if uploaded_file.size > 50 * 1024 * 1024:

            st.error(
                "Fichier trop volumineux (maximum 50 Mo)"
            )

            st.stop()



        try:


            # ==========================
            # Lecture fichier
            # ==========================


            if uploaded_file.name.endswith(".csv"):

                df = pd.read_csv(
                    uploaded_file
                )

            else:

                df = pd.read_excel(
                    uploaded_file
                )



            st.success(
                "Fichier chargé avec succès"
            )


            st.write(
                "Dimensions :",
                df.shape
            )


            st.dataframe(
                df.head()
            )



            # ==========================
            # LANCEMENT ANALYSE
            # ==========================


            if st.button(
                "🚀 Lancer la détection"
            ):



                progress = st.progress(0)

                message = st.empty()



                try:


                    message.info(
                        "Vérification des colonnes..."
                    )


                    progress.progress(
                        10
                    )



                    # Colonnes attendues

                    features = list(
                        scaler.feature_names_in_
                    )



                    missing = list(
                        set(features)
                        -
                        set(df.columns)
                    )



                    if missing:


                        st.error(
                            "Colonnes manquantes :"
                        )


                        st.write(
                            missing
                        )

                        st.stop()



                    # Garder uniquement les features

                    X = df[
                        features
                    ]



                    progress.progress(
                        30
                    )


                    message.info(
                        "Normalisation..."
                    )


                    X_scaled = scaler.transform(
                        X
                    )



                    progress.progress(
                        50
                    )


                    message.info(
                        "Prédiction Deep Learning..."
                    )



                    # Prédiction par batch

                    prediction = model.predict(
                        X_scaled,
                        batch_size=256,
                        verbose=0
                    )



                    progress.progress(
                        80
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



                    # Résultats

                    result = df.copy()


                    result["Prediction"] = labels


                    result["Probabilite (%)"] = (
                        probabilites * 100
                    ).round(2)



                    progress.progress(
                        100
                    )


                    message.success(
                        "Analyse terminée ✅"
                    )



                    st.subheader(
                        "Résultats"
                    )


                    st.dataframe(
                        result
                    )



                    # Statistiques


                    st.subheader(
                        "Statistiques"
                    )


                    st.bar_chart(
                        result["Prediction"]
                        .value_counts()
                    )



                    # Export


                    csv = result.to_csv(
                        index=False
                    )


                    st.download_button(
                        "📥 Télécharger résultat",
                        csv,
                        "IDS2025_resultats.csv",
                        "text/csv"
                    )



                except Exception as e:


                    st.error(
                        f"Erreur analyse : {e}"
                    )



        except Exception as e:


            st.error(
                f"Erreur lecture fichier : {e}"
            )



# ================================
# A PROPOS
# ================================

elif menu == "A propos":


    st.header(
        "Projet IDS2025"
    )


    st.write(
    """
    Système de détection d'intrusion réseau.

    Technologies :

    - Python
    - TensorFlow
    - Keras
    - Streamlit
    - Scikit-Learn

    Modèle :

    MLP Deep Learning
    """
    )



st.divider()

st.caption(
    "Master 2 IA - Détection des Intrusions Réseau"
)
