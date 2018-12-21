#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask
from flask_cors import CORS
from flask import request
import mimetypes
import hashlib
import bz2
import pickle

import json
import numpy as np
import pandas as pd
from fbprophet import Prophet

app = Flask(__name__)
CORS(app)
m = Prophet()

@app.route('/prophet/dataset', methods=["POST"])
def dataset():
    fileid = hashlib.sha224(str(request.data).encode('utf-8')).hexdigest()
    dataFromJson = json.loads(request.data)
    dataNPArray = np.array(dataFromJson)
    df = pd.DataFrame({'ds':dataNPArray[:,0],'y':dataNPArray[:,1]})
    cfile = bz2.BZ2File("./files/"+fileid+".pkl.bz2", 'w')
    pickle.dump(df, cfile)
    return json.dumps({"id":fileid,"import":True})

@app.route('/prophet/<fileid>/fit', methods=["POST"])
def fit(fileid):
    df = pd.read_pickle("./files/"+fileid+".pkl.bz2", compression="bz2")
    m = Prophet()
    m.fit(df)
    cfile = bz2.BZ2File("./models/"+fileid+".pkl.bz2", 'w')
    pickle.dump(m, cfile)
    return json.dumps({"id":fileid,"fit":True})

@app.route('/prophet/<fileid>/predict/<int:periods>', methods=["GET"])
def predict(fileid,periods):
    m = pickle.load(bz2.BZ2File("./models/"+fileid+".pkl.bz2", "r"))
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    return "prophet predict: "+periods

if __name__ == '__main__':
    app.run(debug=True)
