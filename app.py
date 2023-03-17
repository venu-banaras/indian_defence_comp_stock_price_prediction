# import the necessary packages
from flask import Flask, render_template, session, url_for, request
import numpy as np

# for logging output to console
import logging

import tensorflow as tf

keras = tf.keras
from keras.models import load_model
import joblib

# Load the model and the PKL files for prediction
astra_model = load_model("Imp Objects/Models/astra_model.h5")
bdl_model = load_model("Imp Objects/Models/bdl_model.h5")
bel_model = load_model("Imp Objects/Models/bel_model.h5")
beml_model = load_model("Imp Objects/Models/beml_model.h5")
hal_model = load_model("Imp Objects/Models/hal_model.h5")
mazdock_model = load_model("Imp Objects/Models/mazdock_model.h5")
mtar_model = load_model("Imp Objects/Models/mtar_model.h5")
paras_model = load_model("Imp Objects/Models/paras_model.h5")
solar_model = load_model("Imp Objects/Models/solar_model.h5")
zentec_model = load_model("Imp Objects/Models/zentec_model.h5")


init_scaler = joblib.load("Imp Objects\Model objects\scaler.pkl")
init_look_back = joblib.load("Imp Objects\Model objects\look_back.pkl")

scaled_astra_data = joblib.load("Imp Objects/Model objects/scaled_astra_data.pkl")
scaled_bdl_data = joblib.load("Imp Objects/Model objects/scaled_bdl_data.pkl")
scaled_bel_data = joblib.load("Imp Objects/Model objects/scaled_bel_data.pkl")
scaled_beml_data = joblib.load("Imp Objects/Model objects/scaled_beml_data.pkl")
scaled_hal_data = joblib.load("Imp Objects/Model objects/scaled_hal_data.pkl")
scaled_mazdock_data = joblib.load("Imp Objects/Model objects/scaled_mazdock_data.pkl")
scaled_mtar_data = joblib.load("Imp Objects/Model objects/scaled_mtar_data.pkl")
scaled_paras_data = joblib.load("Imp Objects/Model objects/scaled_paras_data.pkl")
scaled_solar_data = joblib.load("Imp Objects/Model objects/scaled_solar_data.pkl")
scaled_zentec_data = joblib.load("Imp Objects/Model objects/scaled_zentec_data.pkl")


# this function is used to predict the stock price for the next n days
# complete description of the function is given in the notebook vedanta_future_pred.ipynb
def return_predictions(model, scaler, num_days, init_scaled_data):

    n_features = init_scaled_data.shape[1]

    forecast = []

    first_eval_batch = init_scaled_data[-init_look_back:]

    current_batch = first_eval_batch.reshape((1, init_look_back, n_features))

    for i in range(num_days):
        current_pred = model.predict(current_batch)[0]

        forecast.append(current_pred)

        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    true_forecast = init_scaler.inverse_transform(forecast)
    return true_forecast


# basic flask code

app = Flask(__name__, template_folder="Templates")

# app.config["SECRET_KEY"] = "mysecretkey"

# retrn the homepage
@app.route("/home", methods=["GET", "POST"])
def home():
    # return "<h1>Flask app is running</h1>"
    return render_template("index.html")


# take in number of days from the user through form in the index.html and return the predicted stock price
# to prediction.html file. and create a dynamic table in the prediction.html file.
@app.route("/stock_pred", methods=["GET", "POST"])
def stock_pred():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        n_days = request.form.get("num")
        company = request.form.get("name")
        # n_days = 2
        n_days = int(n_days)
        company = int(company)

        print("DAYS = ", n_days)
        print("COMPANY = ", company)

        def get_company(comp):
            if comp == 1:
                pred = return_predictions(
                    model=bdl_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_bdl_data
                )
                return pred
            elif comp == 2:
                pred2 = return_predictions(
                    model=hal_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_hal_data
                )
                return pred2
            elif comp == 3:
                pred3 = return_predictions(
                    model=mazdock_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_mazdock_data
                )
                return pred3
            elif comp == 4:
                pred4 = return_predictions(
                    model=mtar_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_mtar_data
                )
                return pred4
            elif comp == 5:
                pred5 = return_predictions(
                    model=solar_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_solar_data
                )
                return pred5
            elif comp == 6:
                pred6 = return_predictions(
                    model=astra_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_astra_data
                )
                return pred6
            elif comp == 7:
                pred7 = return_predictions(
                    model=bel_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_bel_data
                )
                return pred7
            elif comp == 8:
                pred8 = return_predictions(
                    model=paras_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_paras_data
                )
                return pred8
            elif comp == 9:
                pred9 = return_predictions(
                    model=beml_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_beml_data
                )
                return pred9
            elif comp == 10:
                pred10 = return_predictions(
                    model=zentec_model, scaler=init_scaler, num_days=n_days, init_scaled_data=scaled_zentec_data
                )
                return pred10

        # app.logger.info(get_company(company))

        # result = request.form["test"]
        return render_template("prediction.html", results=get_company(comp=company), days=n_days)


if __name__ == "__main__":
    app.run(debug=True)
