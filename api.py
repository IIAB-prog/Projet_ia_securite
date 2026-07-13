# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:32:12 2026

@author: IIAB
"""

import joblib
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/ids_model.keras")

scaler = joblib.load("models/scaler.pkl")

encoder = joblib.load("models/encoder.pkl")


def predict_attack(data):

    data = scaler.transform(data)

    prediction = model.predict(data)

    classe = np.argmax(prediction, axis=1)

    label = encoder.inverse_transform(classe)

    probabilite = np.max(prediction)

    return label[0], probabilite