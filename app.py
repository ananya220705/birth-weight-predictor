from flask import Flask , request,jsonify,render_template
import pandas as pd
import pickle
import os
import os

import os, pickle





app = Flask(__name__)

@app.route('/' , methods=['GET'])
def home() :
    return render_template("index.html")


def get_cleaned_data(form_data):

    gestation=float(form_data['gestation'])
    parity=int(form_data['parity'])
    age=float(form_data['age'])
    height=float(form_data['height'])
    weight=float(form_data['weight'])
    smoke=float(form_data['smoke'])

    cleaned_data={
        "gestation":[gestation],
        "parity":[parity],
        "age":[age],
        "height":[height],
        "weight":[weight],
        "smoke":[smoke]
    }

    return cleaned_data

# define ur endpoint
@app.route("/predict", methods=['POST'] )
def get_prediction():
    # get data from user
    baby_data_form = request.form

    baby_data_cleaned=get_cleaned_data(baby_data_form)


    # convert into dataframe
    baby_df = pd.DataFrame(baby_data_cleaned)

    #load machine learning trained model
    with open("model.pkl",'rb') as obj:
        model = pickle.load(obj)

    # make prediction on user data
    prediction = model.predict(baby_df)
    prediction_value = prediction[0]   # take the first element
    prediction_value = round(float(prediction_value), 2)


    # return response in a json format
    response = {"prediction": prediction_value }

    return render_template("index.html",prediction_value=prediction_value)

if __name__ =='__main__' :
    app.run(debug=True)