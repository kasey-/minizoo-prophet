#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask
from flask_cors import CORS
from flask import request
import mimetypes
import hashlib
import pickle

import pandas as pd
from fbprophet import Prophet

app = Flask(__name__)
CORS(app)
m = Prophet()

def read_dataset(dataset):
    if(dataset.content_type == 'text/csv'):
        return pd.read_csv(dataset)
    elif(
          (dataset.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') or
          (dataset.content_type == 'application/vnd.ms-excel')
        ):
        return pd.read_excel(dataset)
    else:
        return pd.DataFrame({'ds':[],'y':[]})

def sanitize_dataset(df):
    df = df.rename(columns={df.columns[0]:"ds", df.columns[1]:"y"})
    return df

@app.route('/prophet/dataset', methods=["POST"])
def dataset():
    dataset = request.files.get('dataset')
    df = read_dataset(dataset)
    df = sanitize_dataset(df)
    fileid = hashlib.sha224(dataset.filename.encode('utf-8')).hexdigest()
    df.to_pickle("./files/"+fileid+".pkl.compress", compression="gzip")
    if df.empty:
        return "Error at dataset importation"
    return '{"id":"'+fileid+'"}'

@app.route('/prophet/<fileid>/fit', methods=["POST"])
def fit(fileid):
    df = pd.read_pickle("./files/"+fileid+".pkl.compress", compression="gzip")
    m = Prophet()
    m.fit(df)
    with open("./models/"+fileid+".pkl", "wb") as f:
        pickle.dump(m, f)
    return '{"id":"'+fileid+'","fit":True}'

@app.route('/prophet/<fileid>/predict/<int:periods>', methods=["GET"])
def predict(fileid,periods):
    m = pickle.load(open("./models/"+fileid+".pkl", "rb"))
    future = m.make_future_dataframe(periods=periods)
    print(future.tail())
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    return "prophet predict: "+periods

if __name__ == '__main__':
    app.run(debug=True)
