from flask import Flask, render_template, request
from keras.layers import *

import pandas as pd
import numpy as np
import pickle
import joblib
import tensorflow as tf

global models


def init():
    json_file = open('model_train.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_train.h5")
    print("Loaded Model from disk")

    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return loaded_model


app = Flask(__name__, template_folder="template")


@app.route('/')
def home():
    return render_template('index.html')


'''
def valuepredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 13)
    loaded_model = pickle.load(open("ann.pkl", "rb"))
    predict = loaded_model(to_predict)
    return predict[0]'''


@app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features).reshape(1, -1)


    models = init()

    predicted = models.predict(features)

    if predicted == 1:
        res = "likely"
    else:
        res = "not likely"
    # return res
    return render_template('finalprediction.html', prediction=res)


if __name__ == '__main__':
    app.run(debug=True)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
